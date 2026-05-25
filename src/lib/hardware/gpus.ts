export type NvidiaGpuOption = {
  id: string;
  name: string;
  memoryGb: number;
  family: "consumer" | "workstation" | "datacenter";
};

export const nvidiaGpuOptions: NvidiaGpuOption[] = [
  { id: "rtx-5090", name: "GeForce RTX 5090", memoryGb: 32, family: "consumer" },
  { id: "rtx-5080", name: "GeForce RTX 5080", memoryGb: 16, family: "consumer" },
  { id: "rtx-5070-ti", name: "GeForce RTX 5070 Ti", memoryGb: 16, family: "consumer" },
  { id: "rtx-4090", name: "GeForce RTX 4090", memoryGb: 24, family: "consumer" },
  { id: "rtx-4090d", name: "GeForce RTX 4090D", memoryGb: 24, family: "consumer" },
  { id: "rtx-4080-super", name: "GeForce RTX 4080 SUPER", memoryGb: 16, family: "consumer" },
  { id: "rtx-4080", name: "GeForce RTX 4080", memoryGb: 16, family: "consumer" },
  { id: "rtx-4070-ti-super", name: "GeForce RTX 4070 Ti SUPER", memoryGb: 16, family: "consumer" },
  { id: "rtx-4070-ti", name: "GeForce RTX 4070 Ti", memoryGb: 12, family: "consumer" },
  { id: "rtx-3090-ti", name: "GeForce RTX 3090 Ti", memoryGb: 24, family: "consumer" },
  { id: "rtx-3090", name: "GeForce RTX 3090", memoryGb: 24, family: "consumer" },
  { id: "rtx-pro-6000-blackwell", name: "RTX PRO 6000 Blackwell", memoryGb: 96, family: "workstation" },
  { id: "rtx-6000-ada", name: "RTX 6000 Ada", memoryGb: 48, family: "workstation" },
  { id: "rtx-5000-ada", name: "RTX 5000 Ada", memoryGb: 32, family: "workstation" },
  { id: "rtx-4500-ada", name: "RTX 4500 Ada", memoryGb: 24, family: "workstation" },
  { id: "rtx-4000-ada", name: "RTX 4000 Ada", memoryGb: 20, family: "workstation" },
  { id: "b200", name: "B200", memoryGb: 180, family: "datacenter" },
  { id: "h200", name: "H200", memoryGb: 141, family: "datacenter" },
  { id: "h100-sxm", name: "H100 SXM", memoryGb: 80, family: "datacenter" },
  { id: "h100-pcie", name: "H100 PCIe", memoryGb: 80, family: "datacenter" },
  { id: "h800", name: "H800", memoryGb: 80, family: "datacenter" },
  { id: "h20", name: "H20", memoryGb: 96, family: "datacenter" },
  { id: "l40s", name: "L40S", memoryGb: 48, family: "datacenter" },
  { id: "l4", name: "L4", memoryGb: 24, family: "datacenter" },
  { id: "a100-80gb", name: "A100 80GB", memoryGb: 80, family: "datacenter" },
  { id: "a100-40gb", name: "A100 40GB", memoryGb: 40, family: "datacenter" },
  { id: "a800-80gb", name: "A800 80GB", memoryGb: 80, family: "datacenter" },
  { id: "a40", name: "A40", memoryGb: 48, family: "datacenter" },
  { id: "a30", name: "A30", memoryGb: 24, family: "datacenter" },
  { id: "a10", name: "A10", memoryGb: 24, family: "datacenter" },
];