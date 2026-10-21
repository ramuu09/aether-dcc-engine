/**
 * @file aether_dcc.h
 * @brief C/C++ API bindings for AetherDCC Engine v2.0
 * 
 * Provides thread-safe C bindings to the Quaternary Contention ALU,
 * Universal Direct3D 11 BitNet ternary inference, and 64-node Ising solver.
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
 * @brief Returns the human-readable AetherDCC Engine version string.
 */
AETHER_API const char *dcc_sdk_version(void);

/**
 * @brief Performs 1.58-bit ternary neural inference on a 64-pixel (8x8) pattern.
 * 
 * @param pixels Array of 64 normalized floating point values in [-1.0, 1.0].
 * @param class_out Output pointer receiving the predicted class index (0-9) or -1 for blank.
 * @param scores_out Output array of 10 integers receiving raw class scores.
 * @param latency_us Output pointer receiving GPU dispatch duration in microseconds.
 * @return 0 on success, -1 on invalid argument, 1 if GPU divergence occurred.
 */
AETHER_API int dcc_sdk_infer_digit10(
    const float *pixels,
    int *class_out,
    int *scores_out,
    double *latency_us
);

/**
 * @brief Executes a 128-bit SIMD differential contention byte-level cipher block.
 * 
 * @param block 16-byte input plaintext block.
 * @param key 16-byte symmetric key.
 * @param salt 16-byte contention pedestal salt.
 * @param out Output buffer of 16 bytes receiving ciphertext.
 * @param latency_us Output pointer receiving cumulative execution time in microseconds.
 * @return Total bit-error count (0 on ideal execution), or -1 on null pointer.
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
 * 
 * @param neighbors Array of 64*8 neighbor indices (-1 for unused slots).
 * @param coupling Array of 64*8 edge couplings (+1 for ferromagnetic, -1 for antiferromagnetic).
 * @param bias Array of 64 nodal external bias values.
 * @param spins_out Output array of 64 integers receiving final ternary spin states {-1, 0, +1}.
 * @param energy_out Output pointer receiving the final graph cut energy.
 * @param latency_us Output pointer receiving GPU compute dispatch duration in microseconds.
 * @return 0 on success, -1 on invalid graph or null pointer.
 */
AETHER_API int dcc_sdk_solve_ising64(
    const int *neighbors,
    const int *coupling,
    const int *bias,
    int *spins_out,
    int *energy_out,
    double *latency_us
);

#ifdef __cplusplus
}
#endif

#endif /* AETHER_DCC_H */
