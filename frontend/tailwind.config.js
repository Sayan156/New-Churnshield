/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        background: '#09111A',
        card: '#102031',
        border: '#22384D',
        primary: '#D4B483',
        secondary: '#7FB7A3',
        success: '#8FBF9A',
        warning: '#CDA15D',
        danger: '#C96B5C',
        muted: '#9FB1C1',
        ink: '#F5EFE4',
      },
      fontFamily: {
        sans: ['"Avenir Next"', '"Segoe UI"', '"Helvetica Neue"', 'sans-serif'],
        display: ['"Iowan Old Style"', '"Palatino Linotype"', '"Book Antiqua"', 'Georgia', 'serif'],
      },
    },
  },
  plugins: [],
}
