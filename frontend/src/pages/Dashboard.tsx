import { useState, useEffect } from 'react'
import { apiService, type DashboardStats } from '../services/api'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts'

const COLORS = ['#7C3AED', '#06B6D4', '#10B981', '#F59E0B', '#EF4444', '#EC4899']

export default function Dashboard() {
  const [stats, setStats] = useState<DashboardStats | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    loadStats()
  }, [])

  const loadStats = async () => {
    try {
      const response = await apiService.getDashboardStats()
      setStats(response.data)
    } catch (err) {
      setError('Failed to load dashboard stats')
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  const formatPct = (val: number) => `${(val * 100).toFixed(1)}%`
  const leaderboardModels = stats
    ? [...stats.model_comparison].sort((a, b) => {
        if (b.recall !== a.recall) return b.recall - a.recall
        if (b.roc_auc !== a.roc_auc) return b.roc_auc - a.roc_auc
        return b.f1 - a.f1
      })
    : []

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-muted">Loading dashboard...</div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="bg-danger/20 border border-danger/30 rounded-lg p-4 text-danger">
        {error}
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h2 className="text-2xl font-bold text-white">Dashboard Overview</h2>
        <p className="text-muted mt-1">Model performance metrics and comparison</p>
      </div>

      {/* Primary Model Metrics */}
      {stats && (
        <>
          <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
            <MetricCard
              label="ROC-AUC"
              value={formatPct(stats.metrics.model_roc_auc)}
              subtitle="Best Model"
              color="#A78BFA"
            />
            <MetricCard
              label="F1-Score"
              value={formatPct(stats.metrics.model_f1)}
              subtitle="Churn Class"
              color="#22D3EE"
            />
            <MetricCard
              label="Recall"
              value={formatPct(stats.metrics.model_recall)}
              subtitle="Churn Recall"
              color="#10B981"
            />
            <MetricCard
              label="Precision"
              value={formatPct(stats.metrics.model_precision)}
              subtitle="Churn Precision"
              color="#F59E0B"
            />
            <MetricCard
              label="Accuracy"
              value={formatPct(stats.metrics.model_accuracy)}
              subtitle="Overall"
              color="#EC4899"
            />
          </div>

          {/* Model Leaderboard */}
          <div className="glass-card rounded-xl p-6">
            <h3 className="text-lg font-semibold text-white mb-4">Model Leaderboard</h3>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-border">
                    <th className="text-left py-3 px-4 text-muted font-medium">Rank</th>
                    <th className="text-left py-3 px-4 text-muted font-medium">Model</th>
                    <th className="text-right py-3 px-4 text-muted font-medium">Accuracy</th>
                    <th className="text-right py-3 px-4 text-muted font-medium">Precision</th>
                    <th className="text-right py-3 px-4 text-muted font-medium">Recall</th>
                    <th className="text-right py-3 px-4 text-muted font-medium">F1-Score</th>
                    <th className="text-right py-3 px-4 text-muted font-medium">ROC-AUC</th>
                  </tr>
                </thead>
                <tbody>
                  {leaderboardModels.map((model, index) => (
                      <tr
                        key={model.model}
                        className={`
                          border-b border-border/50 last:border-0
                          ${model.model === stats.primary_model.name
                            ? 'bg-primary/10'
                            : ''
                          }
                        `}
                      >
                        <td className="py-3 px-4">
                          {index === 0 ? (
                            <span className="text-yellow-500">🏆</span>
                          ) : (
                            <span className="text-muted">{index + 1}</span>
                          )}
                        </td>
                        <td className="py-3 px-4">
                          <span className={model.model === stats.primary_model.name ? 'text-primary font-medium' : 'text-white'}>
                            {model.model}
                          </span>
                          {model.model === stats.primary_model.name && (
                            <span className="ml-2 text-xs text-primary">Best</span>
                          )}
                        </td>
                        <td className="text-right py-3 px-4 text-white">{formatPct(model.accuracy)}</td>
                        <td className="text-right py-3 px-4 text-white">{formatPct(model.precision)}</td>
                        <td className="text-right py-3 px-4 text-white">{formatPct(model.recall)}</td>
                        <td className="text-right py-3 px-4 text-white">{formatPct(model.f1)}</td>
                        <td className="text-right py-3 px-4 text-white">{formatPct(model.roc_auc)}</td>
                      </tr>
                    ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Visualization */}
          <div className="grid md:grid-cols-2 gap-6">
            <div className="glass-card rounded-xl p-6">
              <h3 className="text-lg font-semibold text-white mb-4">Accuracy Comparison</h3>
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={stats.model_comparison}>
                  <XAxis dataKey="model" tick={{ fill: '#7B82A8', fontSize: 10 }} angle={-45} textAnchor="end" height={80} />
                  <YAxis domain={[0.9, 1]} tick={{ fill: '#7B82A8' }} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#12152C', border: '1px solid #1E2440', borderRadius: '8px' }}
                    labelStyle={{ color: '#C8D0E7' }}
                  />
                  <Bar dataKey="accuracy" radius={[4, 4, 0, 0]}>
                    {stats.model_comparison.map((entry, index) => (
                      <Cell
                        key={`cell-${index}`}
                        fill={entry.model === stats.primary_model.name ? COLORS[0] : '#1E2440'}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div className="glass-card rounded-xl p-6">
              <h3 className="text-lg font-semibold text-white mb-4">Recall Comparison (Churn)</h3>
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={stats.model_comparison}>
                  <XAxis dataKey="model" tick={{ fill: '#7B82A8', fontSize: 10 }} angle={-45} textAnchor="end" height={80} />
                  <YAxis domain={[0.8, 1]} tick={{ fill: '#7B82A8' }} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#12152C', border: '1px solid #1E2440', borderRadius: '8px' }}
                    labelStyle={{ color: '#C8D0E7' }}
                  />
                  <Bar dataKey="recall" radius={[4, 4, 0, 0]}>
                    {stats.model_comparison.map((entry, index) => (
                      <Cell
                        key={`cell-${index}`}
                        fill={entry.model === stats.primary_model.name ? COLORS[0] : '#1E2440'}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </>
      )}
    </div>
  )
}

interface MetricCardProps {
  label: string
  value: string
  subtitle: string
  color: string
}

function MetricCard({ label, value, subtitle, color }: MetricCardProps) {
  return (
    <div className="glass-card rounded-xl p-4 text-center">
      <p className="text-sm text-muted mb-1">{label}</p>
      <p className="text-2xl font-bold" style={{ color }}>{value}</p>
      <p className="text-xs text-muted mt-1">{subtitle}</p>
    </div>
  )
}
