// Flash Attention implementation for OpenCL
// Based on the flash attention algorithm from Dao et al.

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// Flash attention kernel for FP32
__kernel void flash_attn_f32(
    __global const float* Q,    // Query  [batch, n_heads, seq_len, head_dim]
    __global const float* K,    // Key    [batch, n_heads, seq_len, head_dim] 
    __global const float* V,    // Value  [batch, n_heads, seq_len, head_dim]
    __global float* O,          // Output [batch, n_heads, seq_len, head_dim]
    __global const float* mask, // Optional attention mask
    const int batch_size,
    const int n_heads,
    const int seq_len,
    const int head_dim,
    const float scale,
    const int use_mask
) {
    const int head_idx = get_global_id(0);
    const int batch_idx = get_global_id(1);
    const int query_idx = get_global_id(2);
    
    if (head_idx >= n_heads || batch_idx >= batch_size || query_idx >= seq_len) {
        return;
    }
    
    // Calculate base offsets
    const int q_offset = ((batch_idx * n_heads + head_idx) * seq_len + query_idx) * head_dim;
    const int kv_offset_base = (batch_idx * n_heads + head_idx) * seq_len * head_dim;
    const int o_offset = q_offset;
    
    // Load query vector into local memory
    __local float q_vec[128]; // Assuming max head_dim = 128
    for (int d = 0; d < head_dim; d++) {
        q_vec[d] = Q[q_offset + d];
    }
    
    // Initialize attention computation
    float max_score = -INFINITY;
    float sum_exp = 0.0f;
    float output[128] = {0.0f}; // Initialize output accumulator
    
    // First pass: compute max score for numerical stability
    for (int key_idx = 0; key_idx <= query_idx; key_idx++) { // Causal masking
        const int k_offset = kv_offset_base + key_idx * head_dim;
        
        // Compute attention score Q * K^T
        float score = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            score += q_vec[d] * K[k_offset + d];
        }
        score *= scale;
        
        // Apply mask if provided
        if (use_mask && mask) {
            const int mask_offset = (batch_idx * seq_len + query_idx) * seq_len + key_idx;
            score += mask[mask_offset];
        }
        
        max_score = fmax(max_score, score);
    }
    
    // Second pass: compute softmax and weighted sum
    for (int key_idx = 0; key_idx <= query_idx; key_idx++) { // Causal masking
        const int k_offset = kv_offset_base + key_idx * head_dim;
        const int v_offset = kv_offset_base + key_idx * head_dim;
        
        // Recompute attention score
        float score = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            score += q_vec[d] * K[k_offset + d];
        }
        score *= scale;
        
        // Apply mask if provided
        if (use_mask && mask) {
            const int mask_offset = (batch_idx * seq_len + query_idx) * seq_len + key_idx;
            score += mask[mask_offset];
        }
        
        // Compute softmax weight
        float exp_score = exp(score - max_score);
        sum_exp += exp_score;
        
        // Accumulate weighted value
        for (int d = 0; d < head_dim; d++) {
            output[d] += exp_score * V[v_offset + d];
        }
    }
    
    // Normalize and write output
    for (int d = 0; d < head_dim; d++) {
        O[o_offset + d] = output[d] / sum_exp;
    }
}

// Flash attention kernel for FP16
__kernel void flash_attn_f16(
    __global const half* Q,     // Query  [batch, n_heads, seq_len, head_dim]
    __global const half* K,     // Key    [batch, n_heads, seq_len, head_dim]
    __global const half* V,     // Value  [batch, n_heads, seq_len, head_dim]
    __global half* O,           // Output [batch, n_heads, seq_len, head_dim]
    __global const half* mask,  // Optional attention mask
    const int batch_size,
    const int n_heads,
    const int seq_len,
    const int head_dim,
    const float scale,
    const int use_mask
) {
    const int head_idx = get_global_id(0);
    const int batch_idx = get_global_id(1);
    const int query_idx = get_global_id(2);
    
    if (head_idx >= n_heads || batch_idx >= batch_size || query_idx >= seq_len) {
        return;
    }
    
    // Calculate base offsets
    const int q_offset = ((batch_idx * n_heads + head_idx) * seq_len + query_idx) * head_dim;
    const int kv_offset_base = (batch_idx * n_heads + head_idx) * seq_len * head_dim;
    const int o_offset = q_offset;
    
    // Load query vector into local memory (convert to float for computation)
    __local float q_vec[128]; // Assuming max head_dim = 128
    for (int d = 0; d < head_dim; d++) {
        q_vec[d] = vload_half(q_offset + d, Q);
    }
    
    // Initialize attention computation
    float max_score = -INFINITY;
    float sum_exp = 0.0f;
    float output[128] = {0.0f}; // Initialize output accumulator
    
    // First pass: compute max score for numerical stability
    for (int key_idx = 0; key_idx <= query_idx; key_idx++) { // Causal masking
        const int k_offset = kv_offset_base + key_idx * head_dim;
        
        // Compute attention score Q * K^T
        float score = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            score += q_vec[d] * vload_half(k_offset + d, K);
        }
        score *= scale;
        
        // Apply mask if provided
        if (use_mask && mask) {
            const int mask_offset = (batch_idx * seq_len + query_idx) * seq_len + key_idx;
            score += vload_half(mask_offset, mask);
        }
        
        max_score = fmax(max_score, score);
    }
    
    // Second pass: compute softmax and weighted sum
    for (int key_idx = 0; key_idx <= query_idx; key_idx++) { // Causal masking
        const int k_offset = kv_offset_base + key_idx * head_dim;
        const int v_offset = kv_offset_base + key_idx * head_dim;
        
        // Recompute attention score
        float score = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            score += q_vec[d] * vload_half(k_offset + d, K);
        }
        score *= scale;
        
        // Apply mask if provided
        if (use_mask && mask) {
            const int mask_offset = (batch_idx * seq_len + query_idx) * seq_len + key_idx;
            score += vload_half(mask_offset, mask);
        }
        
        // Compute softmax weight
        float exp_score = exp(score - max_score);
        sum_exp += exp_score;
        
        // Accumulate weighted value
        for (int d = 0; d < head_dim; d++) {
            output[d] += exp_score * vload_half(v_offset + d, V);
        }
    }
    
    // Normalize and write output
    for (int d = 0; d < head_dim; d++) {
        vstore_half(output[d] / sum_exp, o_offset + d, O);
    }
}