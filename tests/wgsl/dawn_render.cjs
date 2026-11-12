// Draws one triangle with compiled WGSL on Dawn into a W x H rgba32uint target: the Dawn side of the render
// differential tests.
//
// Usage: node dawn_render.cjs DAWN_NODE VERTEX.wgsl FRAGMENT.wgsl ENTRY W H VERTEX_COUNT TARGET_OUT BINDING...
//
// BINDING is GROUP:BINDING:KIND:FILE as in dawn_run.cjs (visible to the stages the shaders use it in; the pipeline
// layout is derived from the shaders). The target starts as zero; its bytes (row-major, 16 bytes a pixel) go to
// TARGET_OUT, and each storage buffer's first len(FILE) bytes to FILE.out.

const fs = require('fs');

async function main() {
  const [dawnPath, vsPath, fsPath, entry, ws, hs, counts, targetOut, ...bindings] = process.argv.slice(2);
  const { create, globals } = require(dawnPath);
  Object.assign(globalThis, globals);
  const adapter = await create([]).requestAdapter();
  const device = await adapter.requestDevice({});
  let uncaptured = null;
  device.addEventListener('uncapturederror', (e) => { uncaptured = e.error.message; });
  device.pushErrorScope('validation');
  const mod = async (path) => {
    const m = device.createShaderModule({ code: fs.readFileSync(path, 'utf8') });
    for (const msg of (await m.getCompilationInfo()).messages) {
      if (msg.type === 'error') throw new Error(`${path}: ${msg.lineNum}:${msg.linePos}: ${msg.message}`);
    }
    return m;
  };
  const vs = await mod(vsPath);
  const fsm = await mod(fsPath);
  const [w, h] = [Number(ws), Number(hs)];
  const pipeline = device.createRenderPipeline({
    layout: 'auto',
    vertex: { module: vs, entryPoint: entry },
    fragment: { module: fsm, entryPoint: entry, targets: [{ format: 'rgba32uint' }] },
    primitive: { topology: 'triangle-list' },
  });
  const target = device.createTexture({ size: [w, h], format: 'rgba32uint', usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
  const byGroup = new Map();
  const outs = [];
  for (const b of bindings) {
    const [g, n, kind, file] = b.split(':');
    const data = fs.readFileSync(file);
    let size = Math.max(4, Math.ceil(data.length / 4) * 4);
    if (kind === 'uniform') size = Math.max(16, Math.ceil(size / 16) * 16);
    const usage = (kind === 'uniform' ? GPUBufferUsage.UNIFORM : GPUBufferUsage.STORAGE) | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    const buffer = device.createBuffer({ size, usage, mappedAtCreation: true });
    new Uint8Array(buffer.getMappedRange()).set(data);
    buffer.unmap();
    if (!byGroup.has(+g)) byGroup.set(+g, []);
    byGroup.get(+g).push({ binding: +n, resource: { buffer } });
    if (kind !== 'uniform') outs.push({ buffer, file, length: data.length, size });
  }
  const enc = device.createCommandEncoder();
  const pass = enc.beginRenderPass({
    colorAttachments: [{ view: target.createView(), loadOp: 'clear', clearValue: [0, 0, 0, 0], storeOp: 'store' }],
  });
  pass.setPipeline(pipeline);
  for (const [g, entries] of byGroup) {
    pass.setBindGroup(g, device.createBindGroup({ layout: pipeline.getBindGroupLayout(g), entries }));
  }
  pass.draw(Number(counts));
  pass.end();
  const rowBytes = Math.ceil((w * 16) / 256) * 256;
  const tread = device.createBuffer({ size: rowBytes * h, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  enc.copyTextureToBuffer({ texture: target }, { buffer: tread, bytesPerRow: rowBytes }, [w, h]);
  const reads = outs.map((o) => {
    const r = device.createBuffer({ size: o.size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    enc.copyBufferToBuffer(o.buffer, 0, r, 0, o.size);
    return r;
  });
  device.queue.submit([enc.finish()]);
  const err = await device.popErrorScope();
  if (err) throw new Error(`validation: ${err.message}`);
  await tread.mapAsync(GPUMapMode.READ);
  const t = new Uint8Array(tread.getMappedRange());
  const packed = new Uint8Array(w * h * 16);
  for (let y = 0; y < h; y++) packed.set(t.subarray(y * rowBytes, y * rowBytes + w * 16), y * w * 16);
  fs.writeFileSync(targetOut, packed);
  for (let i = 0; i < outs.length; i++) {
    await reads[i].mapAsync(GPUMapMode.READ);
    fs.writeFileSync(outs[i].file + '.out', new Uint8Array(reads[i].getMappedRange()).slice(0, outs[i].length));
  }
  if (uncaptured) throw new Error(`uncaptured: ${uncaptured}`);
  device.destroy();
}

main().then(() => process.exit(0), (e) => { console.error(String(e && e.stack || e)); process.exit(1); });
