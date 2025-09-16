module.exports = {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        midnight: '#0F172A',
        lagoon: '#0EA5E9',
        seafoam: '#2DD4BF',
        sand: '#F1F5F9',
        sunburst: '#FBBF24',
        coral: '#F97316',
        slate: {
          950: '#020617'
        }
      },
      fontFamily: {
        display: ['"Sora"', 'sans-serif'],
        sans: ['"Inter"', 'sans-serif']
      },
      borderRadius: {
        curve: '3.5rem'
      },
      boxShadow: {
        floating: '0 30px 80px -20px rgba(14, 165, 233, 0.35)'
      },
      backgroundImage: {
        'hero-gradient': 'radial-gradient(circle at 20% 20%, rgba(45, 212, 191, 0.35), transparent 55%), radial-gradient(circle at 80% 0%, rgba(251, 191, 36, 0.25), transparent 50%), linear-gradient(135deg, rgba(15, 23, 42, 0.95), rgba(2, 6, 23, 0.97))'
      }
    }
  },
  plugins: []
};
