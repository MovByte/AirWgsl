// Replays a framebuffer-fetch scene on Dawn (its Node.js binding): the Dawn side of tests/fetch/harness.rs.
//
// Usage: node dawn_fetch.cjs DAWN_NODE SCENE.json
//
// The scene names the vertex and fragment WGSL, the bind group layouts (as the compiler described them), the color
// target (format, size, sample count, initial texels or clear value, one-plus-one blending), the topology, a fragment
// buffer shared by every draw, and the steps: each step ends the render pass, copies the target into the snapshot
// texture, begins a pass that loads the target, and issues its draws, each with its bindings (uniform and vertex
// buffers from files, the snapshot, the fragment buffer). The target's bytes (per sample as u32 words for a
// multisampled target) go to `out`, the fragment buffer to `out` + ".buffer". Any validation error, uncaptured error
// or device loss fails the run; the file `out` + ".complete" marks a run whose outputs are written and checked.

const fs = require('fs');

const READ_SAMPLES = `@group(0) @binding(0) var t: texture_multisampled_2d<u32>;
@group(0) @binding(1) var<storage, read_write> o: array<u32>;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let w = textureDimensions(t).x;
    let n = textureNumSamples(t);
    for (var s = 0u; s < n; s = s + 1u) {
        o[(g.y * w + g.x) * n + s] = textureLoad(t, g.xy, i32(s)).x;
    }
}
`;

async function main() {
  const [dawnPath, scenePath] = process.argv.slice(2);
  const s = JSON.parse(fs.readFileSync(scenePath, 'utf8'));
  const { create, globals } = require(dawnPath);
  Object.assign(globalThis, globals);
  const adapter = await create([]).requestAdapter();
  const features = s.features.filter((f) => adapter.features.has(f));
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
  const fsm = await mod(fs.readFileSync(s.fs, 'utf8'));
  const layouts = [];
  for (const g of s.groups) {
    while (layouts.length < g.group) layouts.push(device.createBindGroupLayout({ entries: [] }));
    layouts.push(device.createBindGroupLayout({ entries: g.entries }));
  }
  const target = { format: s.format, writeMask: s.write ? GPUColorWrite.ALL : 0 };
  if (s.blend) {
    const one = { operation: 'add', srcFactor: 'one', dstFactor: 'one' };
    target.blend = { color: one, alpha: one };
  }
  const pipeline = device.createRenderPipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: layouts }),
    vertex: { module: vs, entryPoint: s.entry },
    primitive: { topology: s.topology },
    multisample: { count: s.samples },
    fragment: { module: fsm, entryPoint: s.entry, targets: [target] },
  });
  const size = [s.width, s.height];
  const usage = GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST
    | GPUTextureUsage.TEXTURE_BINDING;
  const color = device.createTexture({ size, format: s.format, sampleCount: s.samples, usage });
  const snapshot = device.createTexture({
    size, format: s.format, sampleCount: s.samples,
    // WebGPU requires RENDER_ATTACHMENT of a multisampled texture.
    usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.TEXTURE_BINDING
      | (s.samples > 1 ? GPUTextureUsage.RENDER_ATTACHMENT : 0),
  });
  if (s.initial) {
    device.queue.writeTexture({ texture: color }, fs.readFileSync(s.initial), { bytesPerRow: s.width * s.bpp }, size);
  }
  const buffer = (data, usage) => {
    let n = Math.max(4, Math.ceil(data.length / 4) * 4);
    if (usage & GPUBufferUsage.UNIFORM) n = Math.max(16, Math.ceil(n / 16) * 16);
    const b = device.createBuffer({ size: n, usage, mappedAtCreation: true });
    new Uint8Array(b.getMappedRange()).set(data);
    b.unmap();
    return b;
  };
  const fragmentBuffer = s.fragmentBuffer
    ? buffer(fs.readFileSync(s.fragmentBuffer), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC) : null;
  const enc = device.createCommandEncoder();
  let first = true;
  for (const step of s.steps) {
    enc.copyTextureToTexture({ texture: color }, { texture: snapshot }, size);
    const attachment = { view: color.createView(), storeOp: 'store' };
    if (first && !s.initial) {
      attachment.loadOp = 'clear';
      attachment.clearValue = s.clear;
    } else {
      attachment.loadOp = 'load';
    }
    if (first && !s.initial) {
      // The clear happens in a pass of its own so that the first snapshot holds the cleared target.
      const clear = enc.beginRenderPass({ colorAttachments: [attachment] });
      clear.end();
      enc.copyTextureToTexture({ texture: color }, { texture: snapshot }, size);
      attachment.loadOp = 'load';
      delete attachment.clearValue;
    }
    first = false;
    const pass = enc.beginRenderPass({ colorAttachments: [attachment] });
    pass.setPipeline(pipeline);
    for (const d of step) {
      const groups = new Map();
      for (const b of d.bindings) {
        let resource;
        if (b.kind === 'snapshot') resource = snapshot.createView();
        else if (b.kind === 'fragment') resource = { buffer: fragmentBuffer };
        else {
          const u = b.kind === 'uniform' ? GPUBufferUsage.UNIFORM : GPUBufferUsage.STORAGE;
          resource = { buffer: buffer(fs.readFileSync(b.file), u) };
        }
        if (!groups.has(b.group)) groups.set(b.group, []);
        groups.get(b.group).push({ binding: b.binding, resource });
      }
      for (let g = 0; g < layouts.length; g++) {
        pass.setBindGroup(g, device.createBindGroup({ layout: layouts[g], entries: groups.get(g) || [] }));
      }
      pass.draw(d.count, d.instances, d.first, d.firstInstance);
    }
    pass.end();
  }
  const reads = [];
  if (s.samples === 1) {
    const row = Math.ceil((s.width * s.bpp) / 256) * 256;
    const b = device.createBuffer({ size: row * s.height, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    enc.copyTextureToBuffer({ texture: color }, { buffer: b, bytesPerRow: row }, size);
    reads.push({ b, row, len: s.width * s.bpp, file: s.out });
  } else {
    const n = s.width * s.height * s.samples * 4;
    const out = device.createBuffer({ size: n, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const cp = device.createComputePipeline({ layout: 'auto', compute: { module: await mod(READ_SAMPLES), entryPoint: 'main' } });
    const cpass = enc.beginComputePass();
    cpass.setPipeline(cp);
    cpass.setBindGroup(0, device.createBindGroup({
      layout: cp.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: color.createView() }, { binding: 1, resource: { buffer: out } }],
    }));
    cpass.dispatchWorkgroups(s.width, s.height, 1);
    cpass.end();
    const b = device.createBuffer({ size: n, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    enc.copyBufferToBuffer(out, 0, b, 0, n);
    reads.push({ b, row: n, len: n, file: s.out });
  }
  if (fragmentBuffer) {
    const n = fragmentBuffer.size;
    const b = device.createBuffer({ size: n, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    enc.copyBufferToBuffer(fragmentBuffer, 0, b, 0, n);
    reads.push({ b, row: n, len: n, file: s.out + '.buffer' });
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
  // Written synchronously, so that it is on disk even if dawn.node ends by a signal afterwards.
  fs.writeFileSync(s.out + '.complete', 'complete\n');
  process.stdout.write('complete\n');
  device.destroy();
}

main().then(() => process.exit(0), (e) => { console.error(String(e && e.stack || e)); process.exit(1); });
