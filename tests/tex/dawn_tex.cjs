// Runs one WGSL compute shader with textures and samplers on Dawn (its Node.js binding): the Dawn side of the texture
// differential tests.
//
// Usage: node dawn_tex.cjs DAWN_NODE PLAN.json
//
// The plan names the shader, its entry point, the workgroup count, and each binding: `uniform`, `storage` and
// `read-only-storage` buffers from a file (storage buffers are written back to FILE.out), textures (format, dimension,
// size, level count, bytes per texel, the view's dimension, whether it is a storage binding, the data file holding every
// level with its layers one after the other, and an optional readback file for level 0), and samplers (a
// GPUSamplerDescriptor), with the bind group layouts of the compiled layout. Any validation error, uncaptured error or
// device loss fails the run.

const fs = require('fs');

async function main() {
  const [dawnPath, planPath] = process.argv.slice(2);
  const plan = JSON.parse(fs.readFileSync(planPath, 'utf8'));
  const { create, globals } = require(dawnPath);
  Object.assign(globalThis, globals);
  const gpu = create([]);
  const adapter = await gpu.requestAdapter();
  if (!adapter) throw new Error('no adapter');
  const features = plan.features.filter((f) => adapter.features.has(f));
  if (features.length !== plan.features.length) throw new Error(`adapter lacks ${plan.features}`);
  const device = await adapter.requestDevice({ requiredFeatures: features });
  let uncaptured = null;
  device.addEventListener('uncapturederror', (e) => { uncaptured = e.error.message; });
  device.pushErrorScope('validation');
  const module = device.createShaderModule({ code: fs.readFileSync(plan.shader, 'utf8') });
  const info = await module.getCompilationInfo();
  for (const m of info.messages) {
    if (m.type === 'error') throw new Error(`shader: ${m.lineNum}:${m.linePos}: ${m.message}`);
  }
  // The executor's layout: the compiled binding layout, group by group.
  const bgls = plan.layouts.map((entries) => device.createBindGroupLayout({ entries }));
  const layout = device.createPipelineLayout({ bindGroupLayouts: bgls });
  const pipeline = device.createComputePipeline({ layout, compute: { module, entryPoint: plan.entry } });
  const byGroup = new Map();
  const outs = [];
  const readbacks = [];
  for (const b of plan.bindings) {
    let resource;
    if (b.kind === 'sampler') {
      resource = device.createSampler(b.desc);
    } else if (b.kind === 'texture') {
      let usage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.COPY_SRC;
      if (b.storage) usage |= GPUTextureUsage.STORAGE_BINDING;
      const tex = device.createTexture({
        format: b.format, dimension: b.dimension, size: b.size, mipLevelCount: b.levels, usage,
      });
      const data = fs.readFileSync(b.file);
      let off = 0;
      for (let l = 0; l < b.levels; l++) {
        const w = Math.max(1, b.size[0] >> l);
        const h = Math.max(1, b.size[1] >> l);
        const layers = b.dimension === '3d' ? Math.max(1, b.size[2] >> l) : b.size[2];
        const n = w * h * layers * b.bpp;
        device.queue.writeTexture({ texture: tex, mipLevel: l }, data.subarray(off, off + n),
          { bytesPerRow: w * b.bpp, rowsPerImage: h }, [w, h, layers]);
        off += n;
      }
      const view = { dimension: b.view };
      if (b.storage) { view.baseMipLevel = 0; view.mipLevelCount = 1; }
      resource = tex.createView(view);
      if (b.readback) readbacks.push({ tex, b });
    } else {
      const data = fs.readFileSync(b.file);
      let size = Math.max(4, Math.ceil(data.length / 4) * 4);
      if (b.kind === 'uniform') size = Math.max(16, Math.ceil(size / 16) * 16);
      const usage = (b.kind === 'uniform' ? GPUBufferUsage.UNIFORM : GPUBufferUsage.STORAGE)
        | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
      const buffer = device.createBuffer({ size, usage, mappedAtCreation: true });
      new Uint8Array(buffer.getMappedRange()).set(data);
      buffer.unmap();
      resource = { buffer };
      if (b.kind !== 'uniform') outs.push({ buffer, file: b.file, length: data.length, size });
    }
    if (!byGroup.has(b.group)) byGroup.set(b.group, []);
    byGroup.get(b.group).push({ binding: b.binding, resource });
  }
  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(pipeline);
  for (const [g, entries] of byGroup) {
    pass.setBindGroup(g, device.createBindGroup({ layout: bgls[g], entries }));
  }
  pass.dispatchWorkgroups(...plan.groups);
  pass.end();
  const reads = outs.map((o) => {
    const r = device.createBuffer({ size: o.size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    enc.copyBufferToBuffer(o.buffer, 0, r, 0, o.size);
    return r;
  });
  const texReads = readbacks.map(({ tex, b }) => {
    const row = Math.ceil(b.size[0] * b.bpp / 256) * 256;
    const layers = b.size[2];
    const r = device.createBuffer({ size: row * b.size[1] * layers, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    enc.copyTextureToBuffer({ texture: tex }, { buffer: r, bytesPerRow: row, rowsPerImage: b.size[1] }, [b.size[0], b.size[1], layers]);
    return { r, row, b };
  });
  device.queue.submit([enc.finish()]);
  const err = await device.popErrorScope();
  if (err) throw new Error(`validation: ${err.message}`);
  for (let i = 0; i < outs.length; i++) {
    await reads[i].mapAsync(GPUMapMode.READ);
    fs.writeFileSync(outs[i].file + '.out', new Uint8Array(reads[i].getMappedRange()).slice(0, outs[i].length));
  }
  for (const { r, row, b } of texReads) {
    await r.mapAsync(GPUMapMode.READ);
    const all = new Uint8Array(r.getMappedRange());
    const tight = b.size[0] * b.bpp;
    const out = new Uint8Array(tight * b.size[1] * b.size[2]);
    for (let z = 0; z < b.size[2]; z++) {
      for (let y = 0; y < b.size[1]; y++) {
        const src = (z * b.size[1] + y) * row;
        out.set(all.subarray(src, src + tight), (z * b.size[1] + y) * tight);
      }
    }
    fs.writeFileSync(b.readback, out);
  }
  if (uncaptured) throw new Error(`uncaptured: ${uncaptured}`);
  device.destroy();
}

main().then(() => process.exit(0), (e) => { console.error(String(e && e.stack || e)); process.exit(1); });
