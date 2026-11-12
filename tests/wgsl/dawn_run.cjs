// Runs one WGSL compute shader on Dawn (its Node.js binding) and writes its buffers back: the Dawn side of the WGSL
// differential tests.
//
// Usage: node dawn_run.cjs DAWN_NODE SHADER.wgsl ENTRY GX,GY,GZ BINDING...
//
// DAWN_NODE is the path of dawn.node. Each BINDING is GROUP:BINDING:KIND:FILE with KIND `uniform`, `storage` or
// `read-only-storage`; the buffer holds FILE's bytes (padded to a multiple of 4, at least 16 for uniforms). The
// shader is dispatched with GX x GY x GZ workgroups; afterwards each storage buffer's first len(FILE) bytes are written
// to FILE.out. Any validation error, uncaptured error or device loss fails the run.

const fs = require('fs');

async function main() {
  const [dawnPath, shaderPath, entry, groups, ...bindings] = process.argv.slice(2);
  const { create, globals } = require(dawnPath);
  Object.assign(globalThis, globals);
  const gpu = create([]);
  const adapter = await gpu.requestAdapter();
  if (!adapter) throw new Error('no adapter');
  const features = [];
  if (adapter.features.has('shader-f16')) features.push('shader-f16');
  const device = await adapter.requestDevice({ requiredFeatures: features });
  let uncaptured = null;
  device.addEventListener('uncapturederror', (e) => { uncaptured = e.error.message; });
  device.pushErrorScope('validation');
  const code = fs.readFileSync(shaderPath, 'utf8');
  const module = device.createShaderModule({ code });
  const info = await module.getCompilationInfo();
  for (const m of info.messages) {
    if (m.type === 'error') throw new Error(`shader: ${m.lineNum}:${m.linePos}: ${m.message}`);
  }
  const pipeline = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: entry } });
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
  const pass = enc.beginComputePass();
  pass.setPipeline(pipeline);
  for (const [g, entries] of byGroup) {
    pass.setBindGroup(g, device.createBindGroup({ layout: pipeline.getBindGroupLayout(g), entries }));
  }
  const [x, y, z] = groups.split(',').map(Number);
  pass.dispatchWorkgroups(x, y, z);
  pass.end();
  const reads = outs.map((o) => {
    const r = device.createBuffer({ size: o.size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    enc.copyBufferToBuffer(o.buffer, 0, r, 0, o.size);
    return r;
  });
  device.queue.submit([enc.finish()]);
  const err = await device.popErrorScope();
  if (err) throw new Error(`validation: ${err.message}`);
  for (let i = 0; i < outs.length; i++) {
    await reads[i].mapAsync(GPUMapMode.READ);
    const bytes = new Uint8Array(reads[i].getMappedRange()).slice(0, outs[i].length);
    fs.writeFileSync(outs[i].file + '.out', bytes);
  }
  if (uncaptured) throw new Error(`uncaptured: ${uncaptured}`);
  device.destroy();
}

main().then(() => process.exit(0), (e) => { console.error(String(e && e.stack || e)); process.exit(1); });
