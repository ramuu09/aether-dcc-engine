/**
 * @file aether_dcc.h
 * @brief C/C++ API bindings for AetherDCC Engine v2.0
 * 
 * Provides thread-safe C bindings to the Quaternary Contention ALU,
 * Universal Direct3D 11 BitNet ternary inference, 64-node Ising solver,
 * high-performance context handles (sub-20 us steady state),
 * and 128-bit SPN block cipher with Strict Avalanche Criterion (SAC) diffusion.
 */

#ifndef AETHER_DCC_H
#define AETHER_DCC_H

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _WIN32
  #ifdef AETHER_DCC_EXPORTS
    #define AETHER_API __declspec(dllexport)
  #else
    #define AETHER_API __declspec(dllimport)
  #endif
#else
  #define AETHER_API
#endif

/**
 * @brief Opaque handle to a persistent, warmed AetherDCC GPU compute context.
 */
typedef struct DccContext DccContext;

/**
 * @brief Returns the human-readable AetherDCC Engine version string.
 */
AETHER_API const char *dcc_sdk_version(void);

/* ========================================================================= */
/* High-Performance Context API (Sub-20 us Steady State Latency)             */
/* ========================================================================= */

/**
 * @brief Creates a persistent DccContext with pre-warmed D3D11 compute pipeline.
 * @return Pointer to DccContext, or NULL if initialization failed.
 */
AETHER_API DccContext *dcc_context_create(void);

/**
 * @brief Destroys and frees a DccContext instance.
 */
AETHER_API void dcc_context_destroy(DccContext *ctx);

/**
 * @brief Performs 1.58-bit ternary neural inference using a persistent context.
 * 
 * @param ctx Context handle.
 * @param pixels Array of 64 normalized floating point values in [-1.0, 1.0].
 * @param class_out Output pointer receiving the predicted class index (0-9) or -1 for blank.
 * @param scores_out Output array of 10 integers receiving raw class scores.
 * @param latency_us Output pointer receiving GPU dispatch duration in microseconds.
 * @return 0 on success, -1 on invalid argument, 1 if GPU divergence occurred.
 */
AETHER_API int dcc_context_infer_digit10(
    DccContext *ctx,
    const float *pixels,
    int *class_out,
    int *scores_out,
    double *latency_us
);

/**
 * @brief Executes a 128-bit SIMD differential contention byte cipher block on a context.
 * 
 * @param ctx Context handle.
 * @param block 16-byte input plaintext block.
 * @param key 16-byte symmetric key.
 * @param salt 16-byte contention pedestal salt.
 * @param out Output buffer of 16 bytes receiving ciphertext.
 * @param latency_us Output pointer receiving cumulative execution time in microseconds.
 * @return Total bit-error count (0 on ideal execution), or -1 on null pointer.
 */
AETHER_API int dcc_context_ghost_cipher(
    DccContext *ctx,
    const unsigned char *block,
    const unsigned char *key,
    const unsigned char *salt,
    unsigned char *out,
    double *latency_us
);

/**
 * @brief Solves a 64-node Ising spin-glass graph cut on a persistent context.
 * 
 * @param ctx Context handle.
 * @param neighbors Array of 64*8 neighbor indices (-1 for unused slots).
 * @param coupling Array of 64*8 edge couplings (+1 for ferromagnetic, -1 for antiferromagnetic).
 * @param bias Array of 64 nodal external bias values.
 * @param spins_out Output array of 64 integers receiving final ternary spin states {-1, 0, +1}.
 * @param energy_out Output pointer receiving the final graph cut energy.
 * @param latency_us Output pointer receiving GPU compute dispatch duration in microseconds.
 * @return 0 on success, -1 on invalid graph or null pointer.
 */
AETHER_API int dcc_context_solve_ising64(
    DccContext *ctx,
    const int *neighbors,
    const int *coupling,
    const int *bias,
    int *spins_out,
    int *energy_out,
    double *latency_us
);

/* ========================================================================= */
/* Warmed Backward-Compatible Endpoints (Zero Initialization Penalty)         */
/* ========================================================================= */

/**
 * @brief Performs 1.58-bit ternary neural inference on a 64-pixel (8x8) pattern.
 */
AETHER_API int dcc_sdk_infer_digit10(
    const float *pixels,
    int *class_out,
    int *scores_out,
    double *latency_us
);

/**
 * @brief Executes a 128-bit SIMD differential contention byte-level cipher block.
 */
AETHER_API int dcc_sdk_ghost_cipher(
    const unsigned char *block,
    const unsigned char *key,
    const unsigned char *salt,
    unsigned char *out,
    double *latency_us
);

/**
 * @brief Evaluates an ALU integration script on the Quaternary CPU contention engine.
 * 
 * @param script_id Script ID: 0 = Tax, 1 = Cipher roundtrip, 2 = Hybrid fold.
 * @param result Output pointer receiving the computed result integer.
 * @return 0 on success, -1 on failure.
 */
AETHER_API int dcc_sdk_eval_script(int script_id, int *result);

/**
 * @brief Solves a 64-node, degree-8 Ising spin-glass graph cut using D3D11 Jacobi relaxation.
 */
AETHER_API int dcc_sdk_solve_ising64(
    const int *neighbors,
    const int *coupling,
    const int *bias,
    int *spins_out,
    int *energy_out,
    double *latency_us
);

/* ========================================================================= */
/* Cryptographic SPN Block Cipher API (Strict Avalanche Criterion Verified)  */
/* ========================================================================= */

/**
 * @brief Encrypts a 16-byte block using a 128-bit Substitution-Permutation Network
 *        with contention-salt key schedule.
 * 
 * @param block 16-byte plaintext input block.
 * @param key 16-byte symmetric key.
 * @param salt 16-byte contention salt.
 * @param out 16-byte ciphertext output buffer.
 * @param rounds Number of SPN rounds (recommended: 4 to 10).
 * @return 0 on success, -1 on null pointer.
 */
AETHER_API int dcc_sdk_ghost_spn_encrypt(
    const unsigned char *block,
    const unsigned char *key,
    const unsigned char *salt,
    unsigned char *out,
    int rounds
);

/**
 * @brief Decrypts a 16-byte block using the inverse 128-bit Substitution-Permutation Network.
 * 
 * @param cipher 16-byte ciphertext input block.
 * @param key 16-byte symmetric key.
 * @param salt 16-byte contention salt.
 * @param out 16-byte plaintext output buffer.
 * @param rounds Number of SPN rounds (must match encryption).
 * @return 0 on success, -1 on null pointer.
 */
AETHER_API int dcc_sdk_ghost_spn_decrypt(
    const unsigned char *cipher,
    const unsigned char *key,
    const unsigned char *salt,
    unsigned char *out,
    int rounds
);

/**
 * @brief Measures the Strict Avalanche Criterion (SAC) diffusion percentage
 *        across all 128 bit flips of a test block.
 * 
 * @param key 16-byte key.
 * @param salt 16-byte salt.
 * @return Avalanche percentage in range [0.0, 100.0] (ideal: ~50.0%), or -1.0 on error.
 */
AETHER_API double dcc_sdk_ghost_sac_score(
    const unsigned char *key,
    const unsigned char *salt
);

#ifdef __cplusplus
}
#endif

#endif /* AETHER_DCC_H */
