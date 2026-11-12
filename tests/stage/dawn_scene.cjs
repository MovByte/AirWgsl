// Replays a render scene of the stage tests on Dawn (its Node.js binding): the Dawn side of tests/stage/harness.rs.
//
// Usage: node dawn_scene.cjs DAWN_NODE SCENE.json
//
// The scene names the vertex and fragment WGSL, the color target (format, size, sample count, clear value), an
// optional depth32float target (cleared to a value, compare "always", written), the topology and the draws; each draw
// lists its bindings (GROUP, BINDING, uniform or storage, the file holding the buffer's bytes) and its index buffer.
// The color target's bytes (row-major; per sample for a 4-sample r32uint target) go to `color`, the depth target's to
// `depthOut`. A 4-sample target is read per sample (as u32 values) with a compute shader. Any validation error, uncaptured error or device loss fails the run; `complete` on standard output marks a run whose outputs are written and checked.

const fs = require('fs');

const READ_SAMPLES = `@group(0) @binding(0) var t: texture_multisampled_2d<u32>;
@group(0) @binding(1) var<storage, read_write> o: array<u32>;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let w = textureDimensions(t).x;
    for (var s = 0u; s < 4u; s = s + 1u) {
        o[(g.y * w + g.x) * 4u + s] = textureLoad(t, g.xy, i32(s)).x;
    }
}
`;

async function main() {
  const [dawnPath, scenePath] = process.argv.slice(2);
  const s = JSON.parse(fs.readFileSync(scenePath, 'utf8'));
  const { create, globals } = require(dawnPath);
  Object.assign(globalThis, globals);
  const adapter = await create([]).requestAdapter();
  const features = [];
  if (adapter.features.has('dual-source-blending')) features.push('dual-source-blending');
  const device = await adapter.requestDevice({ requiredFeatures: features });
  let uncaptured = null;
  device.addEventListener('uncapturederror', (e) => { uncaptured = e.error.message; });
  device.pushErrorScope('validation');
  const mod = async (code) => {
    const m = device.createShaderModule({ code });
    for (const msg of (await m.getCompilationInfo()).messages) {
      if (msg.type === 'error') throw new Error(`${msg.lineNum}:${msg.linePos}: ${msg.message}\n${code}`);
    }
    return m;
  };
  const vs = await mod(fs.readFileSync(s.vs, 'utf8'));
  const fsm = s.fs ? await mod(fs.readFileSync(s.fs, 'utf8')) : null;
  const strip = s.topology.endsWith('strip');
  const indexed = s.draws.find((d) => d.index);
  const primitive = { topology: s.topology };
  if (strip && indexed) primitive.stripIndexFormat = indexed.index.format;
  const desc = {
    layout: 'auto',
    vertex: { module: vs, entryPoint: s.entry },
    primitive,
    multisample: { count: s.samples },
  };
  if (fsm) desc.fragment = { module: fsm, entryPoint: s.entry, targets: [{ format: s.format }] };
  if (s.depth !== null) desc.depthStencil = { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'always' };
  const pipeline = device.createRenderPipeline(desc);
  const size = [s.width, s.height];
  const target = device.createTexture({
    size, format: s.format, sampleCount: s.samples,
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC | GPUTextureUsage.TEXTURE_BINDING,
  });
  const depth = s.depth === null ? null : device.createTexture({
    size, format: 'depth32float', usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
  });
  const buffer = (file, usage) => {
    const data = fs.readFileSync(file);
    let n = Math.max(4, Math.ceil(data.length / 4) * 4);
    if (usage & GPUBufferUsage.UNIFORM) n = Math.max(16, Math.ceil(n / 16) * 16);
    const b = device.createBuffer({ size: n, usage, mappedAtCreation: true });
    new Uint8Array(b.getMappedRange()).set(data);
    b.unmap();
    return b;
  };
  const enc = device.createCommandEncoder();
  const pass = enc.beginRenderPass({
    colorAttachments: [{ view: target.createView(), loadOp: 'clear', clearValue: s.clear, storeOp: 'store' }],
    depthStencilAttachment: depth
      ? { view: depth.createView(), depthLoadOp: 'clear', depthClearValue: s.depth, depthStoreOp: 'store' }
      : undefined,
  });
  pass.setPipeline(pipeline);
  for (const d of s.draws) {
    const groups = new Map();
    for (const b of d.bindings) {
      const usage = b.kind === 'uniform' ? GPUBufferUsage.UNIFORM : GPUBufferUsage.STORAGE;
      if (!groups.has(b.group)) groups.set(b.group, []);
      groups.get(b.group).push({ binding: b.binding, resource: { buffer: buffer(b.file, usage) } });
    }
    for (const [g, entries] of groups) {
      pass.setBindGroup(g, device.createBindGroup({ layout: pipeline.getBindGroupLayout(g), entries }));
    }
    if (d.index) {
      pass.setIndexBuffer(buffer(d.index.file, GPUBufferUsage.INDEX), d.index.format);
      pass.drawIndexed(d.count, d.instances, d.index.first, d.baseVertex, d.firstInstance);
    } else {
      pass.draw(d.count, d.instances, d.first, d.firstInstance);
    }
  }
  pass.end();
  const bpp = { r16uint: 2, r32uint: 4, rgba32uint: 16 }[s.format];
  const reads = [];
  if (s.samples === 1) {
    const row = Math.ceil((s.width * bpp) / 256) * 256;
    const b = device.createBuffer({ size: row * s.height, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    enc.copyTextureToBuffer({ texture: target }, { buffer: b, bytesPerRow: row }, size);
    reads.push({ b, row, len: s.width * bpp, file: s.color });
  } else {
    const n = s.width * s.height * 16;
    const out = device.createBuffer({ size: n, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const cp = device.createComputePipeline({ layout: 'auto', compute: { module: await mod(READ_SAMPLES), entryPoint: 'main' } });
    const cpass = enc.beginComputePass();
    cpass.setPipeline(cp);
    cpass.setBindGroup(0, device.createBindGroup({
      layout: cp.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: target.createView() }, { binding: 1, resource: { buffer: out } }],
    }));
    cpass.dispatchWorkgroups(s.width, s.height, 1);
    cpass.end();
    const b = device.createBuffer({ size: n, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    enc.copyBufferToBuffer(out, 0, b, 0, n);
    reads.push({ b, row: n, len: n, file: s.color });
  }
  if (depth) {
    const row = Math.ceil((s.width * 4) / 256) * 256;
    const b = device.createBuffer({ size: row * s.height, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    enc.copyTextureToBuffer({ texture: depth, aspect: 'depth-only' }, { buffer: b, bytesPerRow: row }, size);
    reads.push({ b, row, len: s.width * 4, file: s.depthOut });
  }
  device.queue.submit([enc.finish()]);
  const err = await device.popErrorScope();
  if (err) throw new Error(`validation: ${err.message}`);
  for (const r of reads) {
    await r.b.mapAsync(GPUMapMode.READ);
    const raw = new Uint8Array(r.b.getMappedRange());
    const rows = raw.length / r.row;
    const packed = new Uint8Array(rows * r.len);
    for (let y = 0; y < rows; y++) packed.set(raw.subarray(y * r.row, y * r.row + r.len), y * r.len);
    fs.writeFileSync(r.file, packed);
  }
  if (uncaptured) throw new Error(`uncaptured: ${uncaptured}`);
  // Every output is written and checked: the harness takes the run as complete from here.
  process.stdout.write('complete\n');
  device.destroy();
}

main().then(() => process.exit(0), (e) => { console.error(String(e && e.stack || e)); process.exit(1); });
