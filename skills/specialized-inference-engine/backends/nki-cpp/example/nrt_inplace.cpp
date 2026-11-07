// Does a NEFF that writes into one of its *input* tensors mutate the caller's device buffer in place?
#include <nrt/nrt.h>
#include <nrt/nrt_experimental.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>
#define CK(x) do { NRT_STATUS s_ = (x); if (s_ != NRT_SUCCESS) { fprintf(stderr, "%s -> %d (%s)\n", #x, (int)s_, nrt_get_status_as_str(s_)); exit(1);} } while (0)
int main(int argc, char** argv) {
  std::ifstream f(argv[1], std::ios::binary); std::vector<char> b((std::istreambuf_iterator<char>(f)), {});
  CK(nrt_init(NRT_FRAMEWORK_TYPE_NO_FW, "", ""));
  nrt_model_t* m; CK(nrt_load(b.data(), b.size(), 0, 1, &m));
  nrt_tensor_info_array_t* info; CK(nrt_get_model_tensor_info(m, &info));
  for (uint64_t i = 0; i < info->tensor_count; i++) printf("  %s %s %zu\n", info->tensor_array[i].usage ? "out" : "in ", info->tensor_array[i].name, info->tensor_array[i].size);
  const int R = 8, D = 256;
  std::vector<float> cache(R * D, -1.f), row(D), out(D), back(R * D);
  for (int i = 0; i < D; i++) row[i] = (float)i;
  nrt_tensor_t *tc, *tr, *to;
  CK(nrt_tensor_allocate(NRT_TENSOR_PLACEMENT_DEVICE, 0, R * D * 4, "cache", &tc));
  CK(nrt_tensor_allocate(NRT_TENSOR_PLACEMENT_DEVICE, 0, D * 4, "row", &tr));
  CK(nrt_tensor_allocate(NRT_TENSOR_PLACEMENT_DEVICE, 0, D * 4, "out", &to));
  CK(nrt_tensor_write(tc, cache.data(), 0, R * D * 4)); CK(nrt_tensor_write(tr, row.data(), 0, D * 4));
  nrt_tensor_set_t *in, *os; CK(nrt_allocate_tensor_set(&in)); CK(nrt_allocate_tensor_set(&os));
  CK(nrt_add_tensor_to_tensor_set(in, "cache", tc)); CK(nrt_add_tensor_to_tensor_set(in, "row", tr));
  CK(nrt_add_tensor_to_tensor_set(os, info->tensor_array[info->tensor_count - 1].name, to));
  CK(nrt_execute(m, in, os));
  CK(nrt_tensor_read(to, out.data(), 0, D * 4)); CK(nrt_tensor_read(tc, back.data(), 0, R * D * 4));
  bool ok_out = true, ok_row3 = true, ok_rest = true;
  for (int i = 0; i < D; i++) { ok_out &= out[i] == 2.f * i; ok_row3 &= back[3 * D + i] == (float)i; }
  for (int r = 0; r < R; r++) if (r != 3) for (int i = 0; i < D; i++) ok_rest &= back[r * D + i] == -1.f;
  printf("output ok=%d  cache row3 updated in place=%d  other rows untouched=%d\n", ok_out, ok_row3, ok_rest);
  nrt_close();
  return !(ok_out && ok_row3 && ok_rest);
}
