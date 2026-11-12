// Records real Metal's pipeline reflection for a list of jobs.
//
// Build: clang -fobjc-arc -O1 -framework Metal -framework Foundation record.m -o record
// Usage: record JOBS.jsonl OUT.jsonl [FIRST]
//
// Each line of JOBS.jsonl is one pipeline to create, as a JSON object:
//
//   id                 a string copied to the output line
//   kind               "compute" or "render"
//   library            path of the metallib the functions come from
//   function           (compute) the kernel
//   vertex             (render) {"function": name} from `library`, or {"source": MSL, "function": name} compiled here
//   fragment           (render, optional) {"function": name} from `library`
//   constants          (optional) [{"index": i, "type": MTLDataType, "hex": little-endian bytes}] for every function
//                      taken from `library`
//   vertex_descriptor  (optional) {"layouts": [{"buffer", "stride", "step_function", "step_rate"}],
//                      "attributes": [{"index", "format", "offset", "buffer"}]}
//   stage_input        (compute, optional) the same shape, for MTLStageInputOutputDescriptor (attribute formats are
//                      MTLAttributeFormat), plus "index_type" and "index_buffer"
//   colors             (render, optional) [[attachment, MTLPixelFormat]]
//   depth, stencil     (render, optional) MTLPixelFormat
//   topology           (render, optional) MTLPrimitiveTopologyClass
//   tessellation       (render, optional) {"partition", "max_factor", "factor_format", "control_point_index_type",
//                      "step_function", "winding", "factor_scale"}
//   dump               (optional) a path: the bitcode of the job's function from `library` (the kernel, or the
//                      fragment function when there is one, else the vertex function) is written there, after
//                      specialization when `constants` is given (`-[_MTLFunction bitcodeData]`, which Metal does not
//                      declare in its headers)
//
// Each output line is a JSON object: id, status ("ok" or "error"), and either "error" (the NSError's domain, code and
// description, or the exception's name and reason) or "stages": one entry per stage with "stage" ("vertex",
// "fragment", "kernel"), "arguments" and "types". Arguments carry every MTLArgument property; nested types
// (MTLStructType, MTLArrayType, MTLPointerType, MTLTextureReferenceType) are entries of "types", numbered in the
// order a depth-first walk of the arguments in order first reaches them (a type reached twice is listed twice), and
// referred to by index. These are the fields and the numbering of the WebMetal wire protocol's Reflection.
//
// A pipeline whose creation fails because the compiler service's connection was interrupted is created again, up to
// three times. A job that crashes the process ends the run; FIRST (default 0) restarts at a later job.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#pragma clang diagnostic ignored "-Wdeprecated-declarations"

@interface NSObject (WMFunctionBitcode)
- (NSData *)bitcodeData;
@end

static void Dump(NSDictionary *job, id<MTLFunction> f) {
    NSString *path = job[@"dump"];
    if (!path || ![(id)f respondsToSelector:@selector(bitcodeData)]) return;
    NSData *bc = [(id)f bitcodeData];
    if (bc.length) [bc writeToFile:path atomically:YES];
}

static const uint32_t kNoType = 0xffffffffu;


@interface WMTypes : NSObject
@property(nonatomic, strong) NSMutableArray *entries;
- (uint32_t)add:(MTLType *)type;
@end

static NSNumber *U(NSUInteger v) { return @(v); }

@implementation WMTypes
- (instancetype)init {
    if ((self = [super init])) _entries = [NSMutableArray array];
    return self;
}

// Index of a new entry for `type` (pre-order), its children numbered after it.
- (uint32_t)add:(MTLType *)type {
    if (!type) return kNoType;
    uint32_t index = (uint32_t)self.entries.count;
    NSMutableDictionary *e = [NSMutableDictionary dictionary];
    [self.entries addObject:e];
    e[@"data_type"] = U(type.dataType);
    switch (type.dataType) {
    case MTLDataTypeStruct: {
        NSMutableArray *members = [NSMutableArray array];
        for (MTLStructMember *m in ((MTLStructType *)type).members) {
            NSMutableDictionary *me = [NSMutableDictionary dictionary];
            me[@"name"] = m.name;
            me[@"offset"] = U(m.offset);
            me[@"data_type"] = U(m.dataType);
            me[@"argument_index"] = U(m.argumentIndex);
            MTLType *t = nil;
            switch (m.dataType) {
            case MTLDataTypeStruct: t = m.structType; break;
            case MTLDataTypeArray: t = m.arrayType; break;
            case MTLDataTypePointer: t = m.pointerType; break;
            case MTLDataTypeTexture: t = m.textureReferenceType; break;
            default: break;
            }
            me[@"type"] = @([self add:t]);
            [members addObject:me];
        }
        e[@"members"] = members;
        break;
    }
    case MTLDataTypeArray: {
        MTLArrayType *a = (MTLArrayType *)type;
        e[@"element_data_type"] = U(a.elementType);
        e[@"array_length"] = U(a.arrayLength);
        e[@"stride"] = U(a.stride);
        e[@"argument_index_stride"] = U(a.argumentIndexStride);
        MTLType *t = nil;
        switch (a.elementType) {
        case MTLDataTypeStruct: t = a.elementStructType; break;
        case MTLDataTypeArray: t = a.elementArrayType; break;
        case MTLDataTypePointer: t = a.elementPointerType; break;
        case MTLDataTypeTexture: t = a.elementTextureReferenceType; break;
        default: break;
        }
        e[@"element_type"] = @([self add:t]);
        break;
    }
    case MTLDataTypePointer: {
        MTLPointerType *p = (MTLPointerType *)type;
        e[@"element_data_type"] = U(p.elementType);
        e[@"access"] = U(p.access);
        e[@"alignment"] = U(p.alignment);
        e[@"data_size"] = U(p.dataSize);
        e[@"element_is_argument_buffer"] = @(p.elementIsArgumentBuffer);
        MTLType *t = nil;
        switch (p.elementType) {
        case MTLDataTypeStruct: t = p.elementStructType; break;
        case MTLDataTypeArray: t = p.elementArrayType; break;
        default: break;
        }
        e[@"element_type"] = @([self add:t]);
        break;
    }
    case MTLDataTypeTexture: {
        MTLTextureReferenceType *t = (MTLTextureReferenceType *)type;
        e[@"texture_data_type"] = U(t.textureDataType);
        e[@"texture_type"] = U(t.textureType);
        e[@"access"] = U(t.access);
        e[@"is_depth_texture"] = @(t.isDepthTexture);
        break;
    }
    default: break;
    }
    return index;
}
@end

static NSDictionary *DescribeArgument(MTLArgument *a, WMTypes *types) {
    NSMutableDictionary *d = [NSMutableDictionary dictionary];
    d[@"name"] = a.name;
    d[@"type"] = U(a.type);
    d[@"access"] = U(a.access);
    d[@"index"] = U(a.index);
    d[@"active"] = @(a.isActive);
    d[@"array_length"] = U(a.arrayLength);
    switch (a.type) {
    case MTLArgumentTypeBuffer:
        d[@"buffer_alignment"] = U(a.bufferAlignment);
        d[@"buffer_data_size"] = U(a.bufferDataSize);
        d[@"buffer_data_type"] = U(a.bufferDataType);
        d[@"buffer_struct_type"] = @([types add:a.bufferStructType]);
        d[@"buffer_pointer_type"] = @([types add:a.bufferPointerType]);
        break;
    case MTLArgumentTypeThreadgroupMemory:
        d[@"threadgroup_memory_alignment"] = U(a.threadgroupMemoryAlignment);
        d[@"threadgroup_memory_data_size"] = U(a.threadgroupMemoryDataSize);
        break;
    case MTLArgumentTypeTexture:
        d[@"texture_type"] = U(a.textureType);
        d[@"texture_data_type"] = U(a.textureDataType);
        d[@"is_depth_texture"] = @(a.isDepthTexture);
        break;
    default: break;
    }
    return d;
}

static NSDictionary *DescribeStage(NSString *stage, NSArray<MTLArgument *> *arguments) {
    WMTypes *types = [WMTypes new];
    NSMutableArray *args = [NSMutableArray array];
    for (MTLArgument *a in arguments) [args addObject:DescribeArgument(a, types)];
    return @{@"stage" : stage, @"arguments" : args, @"types" : types.entries};
}

static NSData *Hex(NSString *s) {
    NSMutableData *d = [NSMutableData dataWithCapacity:s.length / 2];
    for (NSUInteger i = 0; i + 1 < s.length; i += 2) {
        unsigned v = 0;
        sscanf([s substringWithRange:NSMakeRange(i, 2)].UTF8String, "%2x", &v);
        uint8_t b = (uint8_t)v;
        [d appendBytes:&b length:1];
    }
    return d;
}

static NSUInteger N(NSDictionary *d, NSString *k, NSUInteger def) {
    id v = d[k];
    return v ? [v unsignedIntegerValue] : def;
}

@interface WMRecorder : NSObject
@property(nonatomic, strong) id<MTLDevice> device;
@property(nonatomic, strong) NSMutableDictionary<NSString *, id<MTLLibrary>> *libraries;
@end

@implementation WMRecorder
- (id<MTLLibrary>)library:(NSString *)path source:(NSString *)source error:(NSError **)error {
    NSString *key = source ? [@"source:" stringByAppendingString:source] : path;
    id<MTLLibrary> lib = self.libraries[key];
    if (lib) return lib;
    if (source) {
        MTLCompileOptions *o = [MTLCompileOptions new];
        lib = [self.device newLibraryWithSource:source options:o error:error];
    } else {
        lib = [self.device newLibraryWithURL:[NSURL fileURLWithPath:path] error:error];
    }
    if (lib) {
        if (self.libraries.count > 64) [self.libraries removeAllObjects];
        self.libraries[key] = lib;
    }
    return lib;
}

- (id<MTLFunction>)function:(NSString *)name library:(id<MTLLibrary>)lib constants:(NSArray *)constants
                      error:(NSError **)error {
    if (constants.count == 0) {
        id<MTLFunction> f = [lib newFunctionWithName:name];
        if (!f && error)
            *error = [NSError errorWithDomain:@"record" code:1 userInfo:@{NSLocalizedDescriptionKey : @"no such function"}];
        return f;
    }
    MTLFunctionConstantValues *cv = [MTLFunctionConstantValues new];
    for (NSDictionary *c in constants) {
        NSData *bytes = Hex(c[@"hex"]);
        [cv setConstantValue:bytes.bytes type:(MTLDataType)N(c, @"type", 0) atIndex:N(c, @"index", 0)];
    }
    return [lib newFunctionWithName:name constantValues:cv error:error];
}

- (MTLVertexDescriptor *)vertexDescriptor:(NSDictionary *)vd {
    if (!vd) return nil;
    MTLVertexDescriptor *d = [MTLVertexDescriptor vertexDescriptor];
    for (NSDictionary *l in vd[@"layouts"]) {
        MTLVertexBufferLayoutDescriptor *x = d.layouts[N(l, @"buffer", 0)];
        x.stride = N(l, @"stride", 0);
        x.stepFunction = (MTLVertexStepFunction)N(l, @"step_function", 1);
        x.stepRate = N(l, @"step_rate", 1);
    }
    for (NSDictionary *a in vd[@"attributes"]) {
        MTLVertexAttributeDescriptor *x = d.attributes[N(a, @"index", 0)];
        x.format = (MTLVertexFormat)N(a, @"format", 0);
        x.offset = N(a, @"offset", 0);
        x.bufferIndex = N(a, @"buffer", 0);
    }
    return d;
}

- (MTLStageInputOutputDescriptor *)stageInput:(NSDictionary *)sd {
    if (!sd) return nil;
    MTLStageInputOutputDescriptor *d = [MTLStageInputOutputDescriptor stageInputOutputDescriptor];
    for (NSDictionary *l in sd[@"layouts"]) {
        MTLBufferLayoutDescriptor *x = d.layouts[N(l, @"buffer", 0)];
        x.stride = N(l, @"stride", 0);
        x.stepFunction = (MTLStepFunction)N(l, @"step_function", 5);
        x.stepRate = N(l, @"step_rate", 1);
    }
    for (NSDictionary *a in sd[@"attributes"]) {
        MTLAttributeDescriptor *x = d.attributes[N(a, @"index", 0)];
        x.format = (MTLAttributeFormat)N(a, @"format", 0);
        x.offset = N(a, @"offset", 0);
        x.bufferIndex = N(a, @"buffer", 0);
    }
    d.indexType = (MTLIndexType)N(sd, @"index_type", 0);
    d.indexBufferIndex = N(sd, @"index_buffer", 0);
    return d;
}

- (NSDictionary *)run:(NSDictionary *)job {
    NSError *error = nil;
    NSMutableDictionary *out = [NSMutableDictionary dictionaryWithObject:job[@"id"] forKey:@"id"];
    id<MTLLibrary> lib = [self library:job[@"library"] source:nil error:&error];
    if (!lib) return [self fail:out error:error];
    NSArray *constants = job[@"constants"];
    MTLPipelineOption options = MTLPipelineOptionArgumentInfo | MTLPipelineOptionBufferTypeInfo;
    if ([job[@"kind"] isEqual:@"compute"]) {
        id<MTLFunction> f = [self function:job[@"function"] library:lib constants:constants error:&error];
        if (!f) return [self fail:out error:error];
        Dump(job, f);
        MTLComputePipelineDescriptor *d = [MTLComputePipelineDescriptor new];
        d.computeFunction = f;
        d.stageInputDescriptor = [self stageInput:job[@"stage_input"]];
        MTLComputePipelineReflection *r = nil;
        id state = [self.device newComputePipelineStateWithDescriptor:d options:options reflection:&r error:&error];
        if (!state) return [self fail:out error:error];
        out[@"status"] = @"ok";
        out[@"stages"] = @[ DescribeStage(@"kernel", r.arguments ?: @[]) ];
        return out;
    }
    MTLRenderPipelineDescriptor *d = [MTLRenderPipelineDescriptor new];
    NSDictionary *v = job[@"vertex"];
    id<MTLFunction> vf = nil;
    if (v[@"source"]) {
        id<MTLLibrary> vlib = [self library:nil source:v[@"source"] error:&error];
        if (!vlib) return [self fail:out error:error];
        vf = [self function:v[@"function"] library:vlib constants:nil error:&error];
    } else {
        vf = [self function:v[@"function"] library:lib constants:constants error:&error];
    }
    if (!vf) return [self fail:out error:error];
    d.vertexFunction = vf;
    if (job[@"fragment"]) {
        id<MTLFunction> ff = [self function:job[@"fragment"][@"function"] library:lib constants:constants error:&error];
        if (!ff) return [self fail:out error:error];
        d.fragmentFunction = ff;
        Dump(job, ff);
    } else {
        Dump(job, vf);
    }
    d.vertexDescriptor = [self vertexDescriptor:job[@"vertex_descriptor"]];
    for (NSArray *c in job[@"colors"]) d.colorAttachments[[c[0] unsignedIntegerValue]].pixelFormat = [c[1] unsignedIntegerValue];
    d.depthAttachmentPixelFormat = (MTLPixelFormat)N(job, @"depth", 0);
    d.stencilAttachmentPixelFormat = (MTLPixelFormat)N(job, @"stencil", 0);
    d.inputPrimitiveTopology = (MTLPrimitiveTopologyClass)N(job, @"topology", 0);
    NSDictionary *t = job[@"tessellation"];
    if (t) {
        d.tessellationPartitionMode = (MTLTessellationPartitionMode)N(t, @"partition", 0);
        d.maxTessellationFactor = N(t, @"max_factor", 16);
        d.tessellationFactorFormat = (MTLTessellationFactorFormat)N(t, @"factor_format", 0);
        d.tessellationControlPointIndexType = (MTLTessellationControlPointIndexType)N(t, @"control_point_index_type", 0);
        d.tessellationFactorStepFunction = (MTLTessellationFactorStepFunction)N(t, @"step_function", 0);
        d.tessellationOutputWindingOrder = (MTLWinding)N(t, @"winding", 0);
        d.tessellationFactorScaleEnabled = N(t, @"factor_scale", 0) != 0;
    }
    MTLRenderPipelineReflection *r = nil;
    id state = [self.device newRenderPipelineStateWithDescriptor:d options:options reflection:&r error:&error];
    if (!state) return [self fail:out error:error];
    out[@"status"] = @"ok";
    NSMutableArray *stages = [NSMutableArray arrayWithObject:DescribeStage(@"vertex", r.vertexArguments ?: @[])];
    if (job[@"fragment"]) [stages addObject:DescribeStage(@"fragment", r.fragmentArguments ?: @[])];
    out[@"stages"] = stages;
    return out;
}

- (NSDictionary *)fail:(NSMutableDictionary *)out error:(NSError *)error {
    out[@"status"] = @"error";
    out[@"error"] = @{
        @"domain" : error.domain ?: @"",
        @"code" : @(error.code),
        @"description" : error.localizedDescription ?: @""
    };
    return out;
}
@end

int main(int argc, char **argv) {
    @autoreleasepool {
        if (argc < 3) {
            fprintf(stderr, "usage: record JOBS.jsonl OUT.jsonl [FIRST]\n");
            return 2;
        }
        NSUInteger first = argc > 3 ? (NSUInteger)strtoull(argv[3], NULL, 10) : 0;
        NSString *text = [NSString stringWithContentsOfFile:@(argv[1]) encoding:NSUTF8StringEncoding error:nil];
        NSArray *lines = [text componentsSeparatedByString:@"\n"];
        FILE *out = fopen(argv[2], first ? "a" : "w");
        if (!out) return 1;
        WMRecorder *rec = [WMRecorder new];
        rec.device = MTLCreateSystemDefaultDevice();
        rec.libraries = [NSMutableDictionary dictionary];
        NSUInteger n = 0;
        for (NSString *line in lines) {
            if (line.length == 0) continue;
            if (n++ < first) continue;
            @autoreleasepool {
                NSDictionary *job = [NSJSONSerialization JSONObjectWithData:[line dataUsingEncoding:NSUTF8StringEncoding]
                                                                    options:0
                                                                      error:nil];
                NSDictionary *result = nil;
                @try {
                    for (int attempt = 0; attempt < 3; attempt++) {
                        result = [rec run:job];
                        NSString *desc = result[@"error"][@"description"];
                        if (![desc containsString:@"XPC_ERROR_CONNECTION_INTERRUPTED"]) break;
                    }
                } @catch (NSException *e) {
                    result = @{
                        @"id" : job[@"id"],
                        @"status" : @"error",
                        @"error" : @{@"exception" : e.name ?: @"", @"description" : e.reason ?: @""}
                    };
                }
                NSData *json = [NSJSONSerialization dataWithJSONObject:result options:NSJSONWritingSortedKeys error:nil];
                fwrite(json.bytes, 1, json.length, out);
                fputc('\n', out);
                fflush(out);
            }
        }
        fclose(out);
    }
    return 0;
}
