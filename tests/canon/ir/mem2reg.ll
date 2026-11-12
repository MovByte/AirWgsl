target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"

; A variable assigned on both arms of a branch and in a loop; another read before any store (undef).
define i32 @test(i32 %x, i32 %n) {
entry:
  %v = alloca i32, align 4
  %u = alloca i32, align 4
  %c = icmp sgt i32 %x, 10
  br i1 %c, label %then, label %else
then:
  store i32 1, ptr %v, align 4
  br label %join
else:
  store i32 2, ptr %v, align 4
  br label %join
join:
  br label %loop
loop:
  %i = phi i32 [ 0, %join ], [ %i1, %body ]
  %more = icmp ult i32 %i, %n
  br i1 %more, label %body, label %exit
body:
  %a = load i32, ptr %v, align 4
  %b = add i32 %a, %i
  store i32 %b, ptr %v, align 4
  %i1 = add i32 %i, 1
  br label %loop
exit:
  %r = load i32, ptr %v, align 4
  %junk = load i32, ptr %u, align 4
  %m = and i32 %junk, 0
  %s = add i32 %r, %m
  ret i32 %s
}
