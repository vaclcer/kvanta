export function formatBytes(bytes: number, unit: "gib" | "gb" = "gib"): string {
  const divisor = unit === "gib" ? 1024 ** 3 : 1000 ** 3;
  const suffix = unit === "gib" ? "GiB" : "GB";

  return `${(bytes / divisor).toLocaleString(undefined, {
    maximumFractionDigits: 4,
    minimumFractionDigits: bytes / divisor < 10 ? 4 : 2,
  })} ${suffix}`;
}

export function formatInteger(value: number): string {
  return Math.round(value).toLocaleString();
}