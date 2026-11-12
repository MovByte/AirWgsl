target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"

; A callee with a static alloca and two returns, called in a loop; a void callee writing through a pointer.
define internal i32 @pick(i32 %x, i32 %y) {
entry:
  %t = alloca i32, align 4
  store i32 %x, ptr %t, align 4
  %c = icmp slt i32 %x, %y
  br i1 %c, label %less, label %more
less:
  %a = load i32, ptr %t, align 4
  %b = add i32 %a, 1
  ret i32 %b
more:
  %d = mul i32 %y, 3
  ret i32 %d
}

define internal void @put(ptr %p, i32 %v) {
  store i32 %v, ptr %p, align 4
  ret void
}

define i32 @test(i32 %n, ptr addrspace(1) %out) {
entry:
  %cell = alloca i32, align 4
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %next, %loop ]
  %acc = phi i32 [ 0, %entry ], [ %sum, %loop ]
  %r = call i32 @pick(i32 %i, i32 %n)
  %sum = add i32 %acc, %r
  call void @put(ptr %cell, i32 %sum)
  %next = add i32 %i, 1
  %done = icmp uge i32 %next, 8
  br i1 %done, label %exit, label %loop
exit:
  %final = load i32, ptr %cell, align 4
  store i32 %final, ptr addrspace(1) %out, align 4
  ret i32 %sum
}
