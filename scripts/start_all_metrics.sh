#!/usr/bin/env bash
set -euo pipefail

# BIN="./build/llmcpp/train_gpt2_gpu"
BIN="./llmcpp/helloworld"

OUTPUT_PATH="./output/result.csv"
rm "${OUTPUT_PATH}" || true

STEPS="0"

run() {
  local -a metrics=("$@")
  echo "==> ${BIN} ${STEPS} ${metrics[*]}"
  "${BIN}" "${STEPS}" "${metrics[@]}" "-o" "${OUTPUT_PATH}"
}

# -------------------------
# Group 1 — run all together
# -------------------------
grp1=(
  "gpu__time_duration.sum"
  "gpu__time_duration.max"
  "gpc__cycles_elapsed.avg.per_second"
  "gpc__cycles_elapsed.max"
  "sm__cycles_active.max"
)
run "${grp1[@]}"

grp2=(
  "gpu__compute_memory_request_throughput"
  "gpu__compute_memory_throughput"
  "gpu__dram_throughput"
  "dram__throughput"
  
)

# -------------------------
# Group 2 — Sub Group 1
# -------------------------
grp2_sub1=(
  "smsp__inst_executed.sum"
)
run "${grp2_sub1[@]}"

#Group 2 — Sub Group 4
grp2_sub4=(
  "smsp__sass_inst_executed_op_shared_ld.sum"
  "smsp__sass_inst_executed_op_shared_st.sum"
  "smsp__sass_inst_executed_op_global_ld.sum"
  "smsp__sass_inst_executed_op_global_st.sum"
  "smsp__sass_data_bytes_mem_global_op_ld.sum"
  "smsp__sass_data_bytes_mem_global_op_st.sum"
)
run "${grp2_sub4[@]}"


# Group 2 — Sub Group 2
grp2_sub2=(
  "sm__pipe_alu_cycles_active.max"
  "sm__pipe_fma_cycles_active.max"
  "sm__pipe_tensor_cycles_active.max"
  "sm__pipe_shared_cycles_active.max"
)
run "${grp2_sub2[@]}"

# Group 2 — Sub Group 3
grp2_sub3=(
  "sm__sass_inst_executed_op_ldgsts_cache_access.sum"
  "sm__sass_inst_executed_op_ldgsts_cache_bypass.sum"
)
run "${grp2_sub3[@]}"

# Group 2 — Sub Group 4
grp2_sub4=(
  "sm__threads_launched.sum"
  "sm__warps_active.sum"
)
run "${grp2_sub4[@]}"

# -------------------------
# Group 3 — Sub Group 1
# -------------------------
grp3_sub1=(
  "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum"
)
run "${grp3_sub1[@]}"

# Group 3 — Sub Group 2
grp3_sub2=(
  "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum"
)
run "${grp3_sub2[@]}"

# Group 3 — Sub Group 3
grp3_sub3=(
  "l1tex__t_requests_pipe_lsu_mem_global_op_st.sum"
)
run "${grp3_sub3[@]}"

# Group 3 — Sub Group 4
grp3_sub4=(
  "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum"
)
run "${grp3_sub4[@]}"

# Group 3 — Sub Group 5
grp3_sub5=(
  "sm__sass_l1tex_t_requests_pipe_lsu_mem_global_op_ldgsts_cache_access.sum"
  "sm__sass_l1tex_t_sectors_pipe_lsu_mem_global_op_ldgsts_cache_access.sum"
  "sm__sass_l1tex_t_requests_pipe_lsu_mem_global_op_ldgsts_cache_bypass.sum"
  "sm__sass_l1tex_t_sectors_pipe_lsu_mem_global_op_ldgsts_cache_bypass.sum"
)
run "${grp3_sub5[@]}"

# Group 3 — Sub Group 6
grp3_sub6=(
  "lts__t_requests_srcunit_tex_op_read.sum"
  "lts__t_requests_srcunit_tex_op_write.sum"
  "dram__sectors_read.sum"
  "dram__sectors_write.sum"
)
run "${grp3_sub6[@]}"

grp3_sub7=(
  "lts__t_requests_srcunit_l1_op_read.sum"
  "lts__t_requests_srcunit_l1_op_write.sum"
)
run "${grp3_sub7[@]}"

# -------------------------
# Group 4 — Sub Group 1
# -------------------------
grp4_sub1=(
  "smsp__average_warp_latency_per_inst_issued.ratio"
)
run "${grp4_sub1[@]}"

# Group 4 — Sub Group 2
grp4_sub2=(
  "smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio"
  "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio"
)
run "${grp4_sub2[@]}"

# Group 4 — Sub Group 3
grp4_sub3=(
  "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio"
  "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio"
)
run "${grp4_sub3[@]}"

echo "All metric batches completed."
