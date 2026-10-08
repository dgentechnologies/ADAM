/** @type {import('next').NextConfig} */
const nextConfig = {
  /**
   * Static export — the build output in ./out is what Capacitor wraps.
   * Consequences enforced across this app: no API routes, no request-time
   * server components, no middleware, and generateStaticParams on every
   * dynamic route.
   */
  output: 'export',

  /**
   * Emits directory-style routes (/welcome/index.html) so the wrapped app can
   * resolve paths from the local filesystem without a server rewriting URLs.
   */
  trailingSlash: true,

  reactStrictMode: true,
  experimental: { cpus: 2 },

  /** next/image optimisation needs a server; unavailable in a static export. */
  images: {
    unoptimized: true,
  },

  /** Workspace packages ship TS source and are compiled by Next. */
  transpilePackages: ['@adam/ui', '@adam/types'],

  eslint: {
    ignoreDuringBuilds: false,
  },
  typescript: {
    ignoreBuildErrors: false,
  },
};

export default nextConfig;
