import { Outlet, Link, useLocation } from 'react-router-dom'
import { motion } from 'framer-motion'

const navItems = [
  { path: '/dashboard', label: 'Dashboard', icon: '🏠' },
  { path: '/predict', label: 'Predict', icon: '🔮' },
  { path: '/batch', label: 'Batch', icon: '📄' },
  { path: '/arena', label: 'Arena', icon: '📊' },
  { path: '/shap', label: 'SHAP', icon: '🔍' },
]

export default function Layout() {
  const location = useLocation()

  return (
    <div className="min-h-screen bg-background flex flex-col text-ink">
      {/* Header */}
      <header className="sticky top-0 z-50 border-b border-border/80 bg-card/70 backdrop-blur-xl">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            {/* Logo */}
            <div className="flex items-center gap-3">
              <span className="flex h-11 w-11 items-center justify-center rounded-full border border-primary/30 bg-primary/10 text-xl shadow-lg shadow-primary/10">🛡️</span>
              <div>
                <h1 className="font-display text-2xl font-semibold text-gradient">ChurnShield AI</h1>
                <p className="text-[11px] uppercase tracking-[0.24em] text-muted">Real-time churn intelligence</p>
              </div>
            </div>

            {/* Version badge */}
            <div className="hidden sm:flex items-center gap-2 rounded-full border border-primary/25 bg-primary/10 px-3 py-1.5 shadow-md shadow-black/10">
              <span className="text-xs font-semibold uppercase tracking-[0.16em] text-primary">v2.0</span>
              <span className="text-xs text-muted">Stacking (LR Meta)</span>
            </div>
          </div>
        </div>
      </header>

      <div className="flex flex-1">
        {/* Sidebar Navigation */}
        <nav className="hidden w-64 border-r border-border/70 bg-card/30 lg:block">
          <div className="p-4 space-y-2">
            {navItems.map((item) => {
              const isActive = location.pathname === item.path
              return (
                <Link
                  key={item.path}
                  to={item.path}
                  className={`
                    flex items-center gap-3 rounded-xl px-4 py-3 transition-all duration-200
                    ${isActive
                      ? 'panel-outline bg-primary/12 text-ink shadow-lg shadow-black/10'
                      : 'text-muted hover:bg-card/70 hover:text-ink'
                    }
                  `}
                >
                  <span className="text-lg">{item.icon}</span>
                  <span className={isActive ? 'font-medium tracking-[0.04em]' : 'tracking-[0.03em]'}>{item.label}</span>
                </Link>
              )
            })}
          </div>

          {/* Footer */}
          <div className="absolute bottom-0 w-64 border-t border-border px-5 py-4">
            <p className="text-center text-[11px] leading-5 text-muted">
              ChurnShield v2.0
            </p>
            <p className="text-center text-[11px] leading-5 text-muted">
              Django + React
            </p>
          </div>
        </nav>

        {/* Mobile Navigation */}
        <nav className="fixed bottom-0 left-0 right-0 z-50 border-t border-border/80 bg-card/95 backdrop-blur-xl lg:hidden">
          <div className="flex justify-around py-2">
            {navItems.map((item) => {
              const isActive = location.pathname === item.path
              return (
                <Link
                  key={item.path}
                  to={item.path}
                  className={`
                    flex flex-col items-center gap-1 rounded-lg px-4 py-2 transition-all
                    ${isActive
                      ? 'text-primary'
                      : 'text-muted'
                    }
                  `}
                >
                  <span className="text-xl">{item.icon}</span>
                  <span className="text-xs">{item.label}</span>
                </Link>
              )
            })}
          </div>
        </nav>

        {/* Main Content */}
        <main className="flex-1 p-4 sm:p-6 lg:p-8 pb-24 lg:pb-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.45, ease: 'easeOut' }}
            className="mx-auto max-w-7xl"
          >
            <Outlet />
          </motion.div>
        </main>
      </div>
    </div>
  )
}
