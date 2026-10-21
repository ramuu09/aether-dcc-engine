// GhostVault 128-bit SIMD cipher. Dispatch(16, 1, 1), four threads per lane.
// Ops in params[0]: 1 ADD, 2 SUB, 3 XOR, 4 AND, 5 OR.
// Pedestal 4 is added to both banks on every step. Synapses then lock-xadd one bank.

StructuredBuffer<uint> a_bytes : register(t0);
StructuredBuffer<uint> b_bytes : register(t1);
StructuredBuffer<uint> params : register(t2);
RWStructuredBuffer<uint> outbuf : register(u0);
groupshared uint pos;
groupshared uint neg;
groupshared uint carry_bit;
[numthreads(4, 1, 1)]
void cs_main(uint3 gid : SV_GroupID, uint tid : SV_GroupThreadID) {
    uint lane = gid.x;
    uint op = params[0];
    uint bit = 0u;
    uint step = 0u;
    uint prev = 0u;
    uint abit = 0u;
    uint bbit = 0u;
    uint tp = 0u;
    uint tn = 0u;
    uint base = 0u;
    int drive = 0;
    int bdrive = 0;
    if (tid == 0u) {
        carry_bit = (op == 2u) ? 1u : 0u;
    }
    GroupMemoryBarrierWithGroupSync();
    [loop]
    for (bit = 0u; bit < 8u; bit++) {
        drive = 0;
        if (op == 0u) {
            if (bit < 4u) {
                if (tid == 1u) { drive = (bit >= 1u) ? 1 : -1; }
                else if (tid == 2u) { drive = (bit >= 2u) ? 1 : -1; }
                else if (tid == 3u) { drive = (bit >= 3u) ? 1 : -1; }
            }
        } else if (tid == 1u) {
            abit = (a_bytes[lane] >> bit) & 1u;
            drive = (abit == 1u) ? 1 : -1;
        } else if (tid == 2u) {
            bbit = (b_bytes[lane] >> bit) & 1u;
            bdrive = (bbit == 1u) ? 1 : -1;
            if (op == 2u || op == 3u) { bdrive = -bdrive; }
            drive = bdrive;
        } else if (tid == 3u) {
            if (op == 1u || op == 2u) { drive = (carry_bit == 1u) ? 1 : -1; }
            else if (op == 4u) { drive = -1; }
            else if (op == 5u) { drive = 1; }
            else { drive = 0; }
        }
        if (tid == 0u) { pos = 0u; neg = 0u; }
        GroupMemoryBarrierWithGroupSync();
        [loop]
        for (step = 0u; step < 4u; step++) {
            GroupMemoryBarrierWithGroupSync();
            if (tid == 0u) {
                InterlockedAdd(pos, 1u, prev);
                InterlockedAdd(neg, 1u, prev);
            } else if (drive > 0) {
                InterlockedAdd(pos, 1u, prev);
            } else if (drive < 0) {
                InterlockedAdd(neg, 1u, prev);
            }
        }
        GroupMemoryBarrierWithGroupSync();
        if (tid == 0u) {
            InterlockedCompareExchange(pos, 4294967295u, 4294967295u, tp);
            InterlockedCompareExchange(neg, 4294967295u, 4294967295u, tn);
            base = lane * 16u + bit * 2u;
            outbuf[base] = tp;
            outbuf[base + 1u] = tn;
            carry_bit = (tp > tn) ? 1u : 0u;
        }
        GroupMemoryBarrierWithGroupSync();
    }
}
