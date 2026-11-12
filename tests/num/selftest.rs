//! The rounding self-test run on this machine's WebGPU device, and lowered code under the policy it supports.

use crate::gpu::{Gpu, bytes_of};
use airwgsl::num::selftest;

/// Run the probe shader exactly as the executor would: one compute pass over the case buffer.
fn run_probe(cases: &[selftest::Case]) -> Vec<u32> {
    let gpu = Gpu::get();
    let device = &gpu.device;
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("rounding probe"),
        source: wgpu::ShaderSource::Wgsl(selftest::probe_wgsl().into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let input = selftest::probe_input(cases);
    let inbuf = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (input.len() * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    gpu.queue.write_buffer(&inbuf, 0, bytes_of(&input));
    let size = (cases.len() * 4) as u64;
    let out = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let rb = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: inbuf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: out.as_entire_binding() },
        ],
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups((cases.len() as u32).div_ceil(64), 1, 1);
    }
    enc.copy_buffer_to_buffer(&out, 0, &rb, 0, size);
    gpu.queue.submit([enc.finish()]);
    let slice = rb.slice(..);
    slice.map_async(wgpu::MapMode::Read, |r| r.expect("map"));
    device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
    let v: Vec<u32> = slice
        .get_mapped_range()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    rb.unmap();
    v
}

#[test]
fn rounding_self_test_on_this_device() {
    let cases = selftest::cases();
    let results = run_probe(&cases);
    let report = selftest::evaluate(&cases, &results);
    eprintln!("rounding self-test on {}:", Gpu::get().adapter);
    eprintln!(
        "  add/sub {} cases, {} not nearest-even; mul {} cases, {}; fma {} cases, {}; subnormal {} cases, {} differ",
        report.add_sub.cases,
        report.add_sub.failures,
        report.mul.cases,
        report.mul.failures,
        report.fma.cases,
        report.fma.failures,
        report.subnormal.cases,
        report.subnormal.failures
    );
    for (c, got) in report.failures.iter().take(8) {
        eprintln!("  {:?} {:08x} {:08x} {:08x}: got {:08x}, nearest-even {:08x}", c.op, c.a, c.b, c.c, got, c.want);
    }
    match report.native_rounding() {
        Some(n) => eprintln!("  native rounding policy supported: {}", n.evidence()),
        None => eprintln!("  native rounding policy not supported"),
    }
    assert!(report.add_sub.cases > 1000 && report.mul.cases > 1000 && report.fma.cases > 1000);
}
