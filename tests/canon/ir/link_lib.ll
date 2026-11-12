target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"

@k = internal addrspace(2) constant i32 100, align 4

define internal i32 @inner(i32 %x) {
  %k = load i32, ptr addrspace(2) @k, align 4
  %r = add i32 %x, %k
  ret i32 %r
}

define i32 @helper(i32 %x) {
  %r = call i32 @inner(i32 %x)
  %s = shl i32 %r, 1
  ret i32 %s
}
