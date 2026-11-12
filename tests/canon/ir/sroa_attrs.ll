target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"

declare void @llvm.memcpy.p0.p1.i64(ptr noalias writeonly captures(none), ptr addrspace(1) noalias readonly captures(none), i64, i1 immarg)

; A 96-byte structure copied whole from a buffer, the copy's pointers promising 16-byte alignment and 96 dereferenceable
; bytes, then two fields read: the pieces of the copy SROA keeps start 64 and 68 bytes in, where neither promise
; holds.
define i32 @test(ptr addrspace(1) noundef align 16 dereferenceable(96) %src) {
  %s = alloca [96 x i8], align 16
  call void @llvm.memcpy.p0.p1.i64(ptr noundef nonnull align 16 dereferenceable(96) %s, ptr addrspace(1) noundef align 16 dereferenceable(96) %src, i64 96, i1 false)
  %p0 = getelementptr inbounds i8, ptr %s, i64 64
  %a = load i32, ptr %p0, align 16
  %p1 = getelementptr inbounds i8, ptr %s, i64 68
  %b = load i32, ptr %p1, align 4
  %r = add i32 %a, %b
  ret i32 %r
}
