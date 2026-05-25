<script lang="ts">
  import { dev } from "$app/environment";
  import { injectAnalytics } from "@vercel/analytics/sveltekit";
  import { injectSpeedInsights } from "@vercel/speed-insights/sveltekit";
  import "../app.css";

  injectAnalytics({ mode: dev ? "development" : "production" });
  injectSpeedInsights();

  let { children } = $props();
</script>

{@render children()}