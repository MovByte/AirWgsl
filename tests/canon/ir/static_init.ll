target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"

; Function constants and a constructor that derives a constant from them, as the Metal front end writes them.
@fc_i.init = linkonce_odr hidden addrspace(2) externally_initialized constant i32 undef, section "air.fc_initializer", align 4
@fc_b.init = linkonce_odr hidden addrspace(2) externally_initialized constant i8 undef, section "air.fc_initializer", align 1
@derived = internal addrspace(2) global i32 0, align 4
@pred = internal addrspace(2) global i8 0, align 1
@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @init, ptr null }]

declare i1 @air.is_function_constant_defined(ptr addrspace(2) captures(none)) memory(none)
declare i8 @air.normalize_function_constant_predicate.i8(i8) memory(none)

define internal void @init() section "air.static_init" {
  %i = load i32, ptr addrspace(2) @fc_i.init, align 4
  %d = call i1 @air.is_function_constant_defined(ptr addrspace(2) @fc_i.init)
  %v = select i1 %d, i32 %i, i32 7
  %w = mul i32 %v, 2
  store i32 %w, ptr addrspace(2) @derived, align 4
  %b = load i8, ptr addrspace(2) @fc_b.init, align 1
  %bb = and i8 %b, 1
  %p = call i8 @air.normalize_function_constant_predicate.i8(i8 %bb)
  store i8 %p, ptr addrspace(2) @pred, align 1
  ret void
}

define void @test(ptr addrspace(1) %out, ptr addrspace(1) %opt) {
  %v = load i32, ptr addrspace(2) @derived, align 4
  store i32 %v, ptr addrspace(1) %out, align 4
  ret void
}

!air.function_constants = !{!0, !1}
!air.kernel = !{!2}
!0 = !{ptr addrspace(2) @fc_i.init, !"int", !"fc_i", i32 0}
!1 = !{ptr addrspace(2) @fc_b.init, !"bool", !"fc_b", i32 1}
!2 = !{ptr @test, !3, !4}
!3 = !{}
!4 = !{!5, !6}
!5 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"int", !"air.arg_name", !"out"}
!6 = !{i32 1, !"air.function_constant", !7, !"air.buffer", !"air.location_index", i32 5, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"int", !"air.arg_name", !"opt"}
!7 = !{ptr addrspace(2) @pred, !"bool", !"fc_b"}
