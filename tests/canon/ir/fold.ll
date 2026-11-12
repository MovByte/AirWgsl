target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"

@c = internal addrspace(2) constant [3 x i32] [i32 5, i32 6, i32 7], align 4
@g = internal addrspace(1) global i32 9, align 4

; Branches, switches and selects on constants; loads from a constant global (folded) and a mutable one (kept);
; an nsw overflow (poison); a NaN-producing division (its result is a choice: kept); freeze of undef (folded).
define i32 @test(i32 %x, ptr addrspace(1) %out) {
entry:
  %c1 = icmp eq i32 3, 3
  br i1 %c1, label %a, label %b
a:
  %ld = load i32, ptr addrspace(2) getelementptr inbounds ([3 x i32], ptr addrspace(2) @c, i64 0, i64 1), align 4
  %gm = load i32, ptr addrspace(1) @g, align 4
  switch i32 %ld, label %b [ i32 6, label %c
                             i32 5, label %b ]
b:
  %pb = phi i32 [ 0, %entry ], [ 0, %a ], [ 0, %a ]
  ret i32 %pb
c:
  %s = select i1 false, i32 %x, i32 40
  %ov = add nsw i32 2147483647, 1
  %fz = freeze i32 undef
  %nan = fdiv float 0.0, 0.0
  %nb = bitcast float %nan to i32
  store i32 %nb, ptr addrspace(1) %out, align 4
  %t1 = add i32 %s, %gm
  %t2 = add i32 %t1, %fz
  %t3 = add i32 %t2, %ld
  ret i32 %t3
}
