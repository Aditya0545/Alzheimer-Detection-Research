/** @type {import('next').NextConfig} */
const nextConfig = {
  // Enable static exports for better Vercel compatibility
  output: 'standalone',
  
  // Configure image optimization for Vercel
  images: {
    unoptimized: true, // Vercel handles image optimization automatically
  },
  
  // Ensure trailing slashes are handled correctly
  trailingSlash: false,
  
  // Enable react strict mode
  reactStrictMode: true,
  
  // Enable SWC minification
  swcMinify: true,
}

module.exports = nextConfig