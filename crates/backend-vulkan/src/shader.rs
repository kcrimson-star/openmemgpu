use openmemgpu_core::error::CoreError;

pub struct ShaderCompiler;

impl ShaderCompiler {
    /// Compiles an element-wise vector multiplication compute shader into SPIR-V words (u32),
    /// parameterized by the target device's native workgroup size (e.g. 64 or 32).
    pub fn compile_vector_multiply(workgroup_size: u32) -> Result<Vec<u32>, CoreError> {
        let wgsl_source = format!(
            r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;

@compute @workgroup_size({}, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let idx = global_id.x;
    c[idx] = a[idx] * b[idx];
}}
"#,
            workgroup_size
        );

        let module = naga::front::wgsl::parse_str(&wgsl_source)
            .map_err(|e| CoreError::BackendInitFailed(format!("WGSL parse error: {:?}", e)))?;

        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        );

        let module_info = validator.validate(&module).map_err(|e| {
            CoreError::BackendInitFailed(format!("Shader validation error: {:?}", e))
        })?;

        let options = naga::back::spv::Options {
            lang_version: (1, 3),
            ..Default::default()
        };

        let spv_words =
            naga::back::spv::write_vec(&module, &module_info, &options, None).map_err(|e| {
                CoreError::BackendInitFailed(format!("SPIR-V generation error: {:?}", e))
            })?;

        Ok(spv_words)
    }

    /// Compiles an element-wise vector scale-bias compute shader into SPIR-V words (u32):
    /// out[idx] = in[idx] * 1.5 + 0.5
    pub fn compile_vector_scale_bias(workgroup_size: u32) -> Result<Vec<u32>, CoreError> {
        let wgsl_source = format!(
            r#"
@group(0) @binding(0) var<storage, read> in_buf: array<f32>;
@group(0) @binding(1) var<storage, read_write> out_buf: array<f32>;

@compute @workgroup_size({}, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let idx = global_id.x;
    out_buf[idx] = in_buf[idx] * 1.5 + 0.5;
}}
"#,
            workgroup_size
        );

        let module = naga::front::wgsl::parse_str(&wgsl_source)
            .map_err(|e| CoreError::BackendInitFailed(format!("WGSL parse error: {:?}", e)))?;

        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        );

        let module_info = validator.validate(&module).map_err(|e| {
            CoreError::BackendInitFailed(format!("Shader validation error: {:?}", e))
        })?;

        let options = naga::back::spv::Options {
            lang_version: (1, 3),
            ..Default::default()
        };

        let spv_words =
            naga::back::spv::write_vec(&module, &module_info, &options, None).map_err(|e| {
                CoreError::BackendInitFailed(format!("SPIR-V generation error: {:?}", e))
            })?;

        Ok(spv_words)
    }

    /// Compiles a 2D Matrix Multiplication (GEMM: C = A * B) compute shader into SPIR-V.
    /// A is (M x K), B is (K x N), C is (M x N).
    /// Workgroup size is typically (16, 16, 1) or (8, 8, 1).
    pub fn compile_gemm(tile_x: u32, tile_y: u32) -> Result<Vec<u32>, CoreError> {
        let wgsl_source = format!(
            r#"
struct GemmParams {{
    M: u32,
    K: u32,
    N: u32,
    pad: u32,
}};

@group(0) @binding(0) var<storage, read> a_mat: array<f32>;
@group(0) @binding(1) var<storage, read> b_mat: array<f32>;
@group(0) @binding(2) var<storage, read_write> c_mat: array<f32>;
@group(0) @binding(3) var<uniform> params: GemmParams;

@compute @workgroup_size({}, {}, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let col = global_id.x;
    let row = global_id.y;

    if (row < params.M && col < params.N) {{
        var acc: f32 = 0.0;
        let k_dim = params.K;
        for (var k: u32 = 0u; k < k_dim; k = k + 1u) {{
            let a_val = a_mat[row * k_dim + k];
            let b_val = b_mat[k * params.N + col];
            acc = acc + a_val * b_val;
        }}
        c_mat[row * params.N + col] = acc;
    }}
}}
"#,
            tile_x, tile_y
        );

        let module = naga::front::wgsl::parse_str(&wgsl_source)
            .map_err(|e| CoreError::BackendInitFailed(format!("WGSL parse error: {:?}", e)))?;

        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        );

        let module_info = validator.validate(&module).map_err(|e| {
            CoreError::BackendInitFailed(format!("Shader validation error: {:?}", e))
        })?;

        let options = naga::back::spv::Options {
            lang_version: (1, 3),
            ..Default::default()
        };

        let spv_words =
            naga::back::spv::write_vec(&module, &module_info, &options, None).map_err(|e| {
                CoreError::BackendInitFailed(format!("SPIR-V generation error: {:?}", e))
            })?;

        Ok(spv_words)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compile_vector_multiply_shader() {
        // Compile for workgroup size 64 (Renoir / Navi 10 default)
        let words_64 =
            ShaderCompiler::compile_vector_multiply(64).expect("Should compile WGSL to SPIR-V");
        assert!(!words_64.is_empty());
        assert_eq!(
            words_64[0], 0x07230203,
            "Must start with SPIR-V magic header"
        );

        // Compile for workgroup size 32
        let words_32 =
            ShaderCompiler::compile_vector_multiply(32).expect("Should compile WGSL to SPIR-V");
        assert!(!words_32.is_empty());
        assert_eq!(
            words_32[0], 0x07230203,
            "Must start with SPIR-V magic header"
        );
    }

    #[test]
    fn test_compile_gemm_shader() {
        let words =
            ShaderCompiler::compile_gemm(16, 16).expect("Should compile GEMM WGSL to SPIR-V");
        assert!(!words.is_empty());
        assert_eq!(words[0], 0x07230203, "Must start with SPIR-V magic header");
    }
}
