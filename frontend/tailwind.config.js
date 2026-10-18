/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        sidebarBg: '#0F2E28',
        sidebarActive: '#1A4A3E',
        accentTeal: '#2DD4BF',
        heatCool: '#2B6CB0',
        heatMid: '#D9A441',
        heatHot: '#E0472B',
        success: '#2F9E5B',
        warning: '#D9A441',
        danger: '#E0472B',
        appBg: '#F5F7FA'
      },
      boxShadow: {
        card: '0 10px 28px rgba(15,46,40,.07)',
        soft: '0 6px 18px rgba(15,46,40,.09)'
      }
    }
  },
  plugins: []
}
