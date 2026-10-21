// AetherDCC 1.58-bit digit net. One dispatch, cs_5_0.
// Layer 1: 64 pixels -> 16 hidden neurons. W == 0 skips InterlockedAdd.
// Layer 2: 16 hidden centered counts -> 10 digit scores.
// An all -1 canvas produces centered hidden 0 and class scores 0.

StructuredBuffer<float> pixels : register(t0);
StructuredBuffer<int> w1 : register(t1);
StructuredBuffer<int> w2 : register(t2);
RWStructuredBuffer<int4> outbuf : register(u0);

groupshared uint p[16];
groupshared uint n[16];
groupshared int hid[16];

[numthreads(64, 1, 1)]
void cs_main(uint tid : SV_GroupThreadID) {
    uint h = 0u;
    uint j = 0u;
    uint i = 0u;
    uint prev = 0u;
    int w = 0;
    int hv = 0;
    int sumw = 0;
    int centered = 0;
    int ternary = 0;
    int pos = 0;
    int neg = 0;
    int toward_pos = 0;

    if (tid < 16u) {
        p[tid] = 0u;
        n[tid] = 0u;
        hid[tid] = 0;
    }
    GroupMemoryBarrierWithGroupSync();

    [loop]
    for (h = 0u; h < 16u; h++) {
        w = w1[h * 64u + tid];
        if (w != 0) {
            toward_pos = 0;
            if (w > 0 && pixels[tid] > 0.0) toward_pos = 1;
            if (w < 0 && pixels[tid] < 0.0) toward_pos = 1;
            if (toward_pos == 1) InterlockedAdd(p[h], 1u, prev);
            else InterlockedAdd(n[h], 1u, prev);
        }
    }
    GroupMemoryBarrierWithGroupSync();

    if (tid < 16u) {
        sumw = 0;
        [loop]
        for (i = 0u; i < 64u; i++) {
            w = w1[tid * 64u + i];
            if (w != 0) sumw += w;
        }
        centered = (int)p[tid] - (int)n[tid] + sumw;
        ternary = 0;
        if (centered > 4) ternary = 1;
        else if (centered < 4) ternary = -1;
        hid[tid] = centered;
        outbuf[tid] = int4((int)p[tid], (int)n[tid], centered, ternary);
    }
    GroupMemoryBarrierWithGroupSync();

    if (tid < 10u) {
        pos = 0;
        neg = 0;
        [loop]
        for (j = 0u; j < 16u; j++) {
            w = w2[tid * 16u + j];
            hv = hid[j];
            if (w != 0 && hv != 0) {
                if (w > 0) {
                    if (hv > 0) pos += hv;
                    else neg += -hv;
                } else {
                    if (hv > 0) neg += hv;
                    else pos += -hv;
                }
            }
        }
        outbuf[16u + tid] = int4(pos, neg, pos - neg, 0);
    }
}
