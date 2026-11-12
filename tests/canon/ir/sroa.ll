target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"

%pair = type { i32, float, <2 x i16> }

@table = internal addrspace(2) constant %pair { i32 7, float 2.5, <2 x i16> <i16 3, i16 4> }, align 4

declare void @llvm.memcpy.p0.p2.i64(ptr noalias writeonly, ptr addrspace(2) noalias readonly, i64, i1 immarg)
declare void @llvm.memset.p0.i64(ptr writeonly, i8, i64, i1 immarg)
declare void @llvm.lifetime.start.p0(i64 immarg, ptr captures(none))
declare void @llvm.lifetime.end.p0(i64 immarg, ptr captures(none))

; A struct copied from a constant global, then read and written field by field; an array cleared and indexed by a
; constant; a dynamically indexed array that stays in memory.
define i32 @test(i32 %k) {
  %s = alloca %pair, align 4
  %arr = alloca [4 x i32], align 4
  %dyn = alloca [4 x i32], align 4
  call void @llvm.lifetime.start.p0(i64 12, ptr %s)
  call void @llvm.memcpy.p0.p2.i64(ptr %s, ptr addrspace(2) @table, i64 12, i1 false)
  %f0 = getelementptr inbounds %pair, ptr %s, i64 0, i32 0
  %a = load i32, ptr %f0, align 4
  %f1 = getelementptr inbounds %pair, ptr %s, i64 0, i32 1
  %b = load float, ptr %f1, align 4
  %bi = fptosi float %b to i32
  %f2 = getelementptr inbounds %pair, ptr %s, i64 0, i32 2
  %v = load <2 x i16>, ptr %f2, align 4
  %e = extractelement <2 x i16> %v, i32 1
  %ez = zext i16 %e to i32
  store i32 %k, ptr %f0, align 4
  %a2 = load i32, ptr %f0, align 4
  call void @llvm.lifetime.end.p0(i64 12, ptr %s)
  call void @llvm.memset.p0.i64(ptr %arr, i8 0, i64 16, i1 false)
  %e2 = getelementptr inbounds [4 x i32], ptr %arr, i64 0, i64 2
  store i32 %a2, ptr %e2, align 4
  %e3 = getelementptr inbounds [4 x i32], ptr %arr, i64 0, i64 3
  %z = load i32, ptr %e3, align 4
  %x2 = load i32, ptr %e2, align 4
  %d0 = getelementptr inbounds [4 x i32], ptr %dyn, i64 0, i64 0
  store i32 11, ptr %d0, align 4
  %d1 = getelementptr inbounds [4 x i32], ptr %dyn, i64 0, i64 1
  store i32 12, ptr %d1, align 4
  %ki = and i32 %k, 1
  %kz = zext i32 %ki to i64
  %dk = getelementptr inbounds [4 x i32], ptr %dyn, i64 0, i64 %kz
  %dv = load i32, ptr %dk, align 4
  %r1 = add i32 %a, %bi
  %r2 = add i32 %r1, %ez
  %r3 = add i32 %r2, %z
  %r4 = add i32 %r3, %x2
  %r5 = add i32 %r4, %dv
  ret i32 %r5
}
