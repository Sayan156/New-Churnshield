import { useState, useEffect } from 'react'
import { apiService } from '../services/api'
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, Cell } from 'recharts'

const COLORS = ['#7C3AED', '#06B6D4', '#10B981', '#F59E0B', '#EF4444', '#EC4899']

export default function Arena() {
  const [models, setModels] = useState<any[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    loadModels()
  }, [])

  const loadModels = async () => {
    try {
      const response = await apiService.getDashboardStats()
      setModels(response.data.model_comparison)
    } catch (err) {
      setError('Failed to load model data')
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-muted">Loading model arena...</div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="bg-danger/20 border border-danger/30 rounded-xl p-4 text-danger">
        {error}
      </div>
    )
  }

  const metrics = [
    { key: 'accuracy', label: 'Accuracy', domain: [0.9, 1] as [number, number] },
    { key: 'precision', label: 'Precision', domain: [0.7, 1] as [number, number] },
    { key: 'recall', label: 'Recall', domain: [0.8, 1] as [number, number] },
    { key: 'f1', label: 'F1-Score', domain: [0.8, 1] as [number, number] },
    { key: 'roc_auc', label: 'ROC-AUC', domain: [0.97, 1] as [number, number] },
  ]
  const rankedModels = [...models].sort((a, b) => {
    if (b.recall !== a.recall) return b.recall - a.recall
    if (b.roc_auc !== a.roc_auc) return b.roc_auc - a.roc_auc
    return b.f1 - a.f1
  })

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h2 className="text-2xl font-bold text-white">Model Arena</h2>
        <p className="text-muted mt-1">Head-to-head comparison of all models</p>
      </div>

      {/* Per-Metric Comparison */}
      <div className="grid md:grid-cols-2 lg:grid-cols-5 gap-4">
        {metrics.map((metric) => {
          const data = rankedModels.map((m) => ({
            name: m.model.length > 15 ? m.model.substring(0, 15) + '...' : m.model,
            fullName: m.model,
            value: m[metric.key] * 100,
            isBest: m.model === 'Stacking (LR Meta)',
          }))

          return (
            <div key={metric.key} className="glass-card rounded-xl p-4">
              <h3 className="text-sm font-semibold text-muted mb-3 text-center">{metric.label}</h3>
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={data} layout="vertical" margin={{ left: 20 }}>
                  <XAxis type="number" domain={metric.domain.map((d) => d * 100)} hide />
                  <YAxis
                    dataKey="name"
                    type="category"
                    tick={{ fill: '#7B82A8', fontSize: 9 }}
                    width={100}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#12152C',
                      border: '1px solid #1E2440',
                      borderRadius: '8px',
                    }}
                    labelStyle={{ color: '#C8D0E7' }}
                    formatter={(value: number) => [`${value.toFixed(2)}%`, metric.label]}
                    labelFormatter={(label) => data.find((d) => d.name === label)?.fullName || label}
                  />
                  <Bar dataKey="value" barSize={20} radius={[0, 4, 4, 0]}>
                    {data.map((entry, index) => (
                      <Cell
                        key={`cell-${index}`}
                        fill={entry.isBest ? COLORS[0] : '#1E2440'}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          )
        })}
      </div>

      {/* Detailed Comparison Table */}
      <div className="glass-card rounded-xl p-6">
        <h3 className="text-lg font-semibold text-white mb-4">Detailed Comparison</h3>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-border">
                <th className="text-left py-3 px-4 text-muted font-medium">Model</th>
                <th className="text-right py-3 px-4 text-muted font-medium">Accuracy</th>
                <th className="text-right py-3 px-4 text-muted font-medium">Precision</th>
                <th className="text-right py-3 px-4 text-muted font-medium">Recall</th>
                <th className="text-right py-3 px-4 text-muted font-medium">F1-Score</th>
                <th className="text-right py-3 px-4 text-muted font-medium">ROC-AUC</th>
                <th className="text-center py-3 px-4 text-muted font-medium">Rank</th>
              </tr>
            </thead>
            <tbody>
              {rankedModels.map((model, index) => (
                  <tr
                    key={model.model}
                    className={`border-b border-border/50 last:border-0 ${
                      model.model === 'Stacking (LR Meta)' ? 'bg-primary/10' : ''
                    }`}
                  >
                    <td className="py-3 px-4">
                      <span className={model.model === 'Stacking (LR Meta)' ? 'text-primary font-medium' : 'text-white'}>
                        {model.model}
                      </span>
                      {model.model === 'Stacking (LR Meta)' && (
                        <span className="ml-2 text-xs text-primary">🏆 Best</span>
                      )}
                    </td>
                    <td className="text-right py-3 px-4 text-white">{(model.accuracy * 100).toFixed(2)}%</td>
                    <td className="text-right py-3 px-4 text-white">{(model.precision * 100).toFixed(2)}%</td>
                    <td className="text-right py-3 px-4 text-white">{(model.recall * 100).toFixed(2)}%</td>
                    <td className="text-right py-3 px-4">
                      <span className={model.model === 'Stacking (LR Meta)' ? 'text-primary font-bold' : 'text-white'}>
                        {(model.f1 * 100).toFixed(2)}%
                      </span>
                    </td>
                    <td className="text-right py-3 px-4 text-white">{(model.roc_auc * 100).toFixed(2)}%</td>
                    <td className="text-center py-3 px-4">
                      {index === 0 ? (
                        <span className="text-2xl">🥇</span>
                      ) : index === 1 ? (
                        <span className="text-2xl">🥈</span>
                      ) : index === 2 ? (
                        <span className="text-2xl">🥉</span>
                      ) : (
                        <span className="text-muted">{index + 1}</span>
                      )}
                    </td>
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Radar Chart Placeholder */}
      <div className="glass-card rounded-xl p-6">
        <h3 className="text-lg font-semibold text-white mb-4">Model Performance Radar</h3>
        <div className="h-80 flex items-center justify-center text-muted">
          <p>Radar chart visualization (coming soon)</p>
        </div>
      </div>
    </div>
  )
}
