target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"

define internal i32 @even(i32 %x) {
  %z = icmp eq i32 %x, 0
  br i1 %z, label %yes, label %no
yes:
  ret i32 1
no:
  %y = sub i32 %x, 1
  %r = call i32 @odd(i32 %y)
  ret i32 %r
}

define internal i32 @odd(i32 %x) {
  %z = icmp eq i32 %x, 0
  br i1 %z, label %yes, label %no
yes:
  ret i32 0
no:
  %y = sub i32 %x, 1
  %r = call i32 @even(i32 %y)
  ret i32 %r
}

define i32 @test(i32 %x) {
  %r = call i32 @even(i32 %x)
  ret i32 %r
}
