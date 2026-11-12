target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"

declare void @llvm.lifetime.start.p0(i64 immarg, ptr captures(none))
declare void @llvm.lifetime.end.p0(i64 immarg, ptr captures(none))

; LangRef "freeze": all uses of one freeze see the same value, so x + x is even.
define i32 @freeze_even() {
  %x = freeze i32 undef
  %y = add i32 %x, %x
  ret i32 %y
}

; Two uses of undef are independent: undef + undef is undef in every bit.
define i32 @undef_twice() {
  %y = add i32 undef, undef
  ret i32 %y
}

; An uninitialized stack object reads as undef; and with 0 is 0.
define i32 @uninit_and_zero() {
  %a = alloca i32, align 4
  %v = load i32, ptr %a, align 4
  %r = and i32 %v, 0
  ret i32 %r
}

; An object a lifetime.start names starts dead: loading it first is undefined behavior.
define i32 @dead_before_start() {
  %a = alloca i32, align 4
  %v = load i32, ptr %a, align 4
  call void @llvm.lifetime.start.p0(i64 4, ptr %a)
  ret i32 %v
}

; getelementptr inbounds past the end (not one past) is poison; one past is fine.
define i64 @gep_bounds(i64 %k) {
  %a = alloca [4 x i32], align 4
  %p = getelementptr inbounds [4 x i32], ptr %a, i64 0, i64 %k
  %i = ptrtoint ptr %p to i64
  %b = ptrtoint ptr %a to i64
  %d = sub i64 %i, %b
  ret i64 %d
}

; fptosi of a value that does not fit is poison.
define i32 @fptosi_range(float %x) {
  %r = fptosi float %x to i32
  ret i32 %r
}

; 0 * inf is NaN: the preferred NaN with its sign a choice; under nnan, poison.
define float @nan_result(float %x) {
  %r = fmul float %x, 0x7FF0000000000000
  ret float %r
}

define float @nan_nnan(float %x) {
  %r = fmul nnan float %x, 0x7FF0000000000000
  ret float %r
}

; select on undef picks one operand: the bits where both agree are defined.
define i32 @select_undef() {
  %r = select i1 undef, i32 12, i32 13
  ret i32 %r
}

; A vector bitcast keeps poison to the lanes it came from.
define <2 x i32> @bitcast_lanes() {
  %v = bitcast <4 x i16> <i16 1, i16 2, i16 poison, i16 4> to <2 x i32>
  ret <2 x i32> %v
}
