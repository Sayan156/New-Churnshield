import { useState } from 'react'
import { useForm } from 'react-hook-form'
import { apiService, type PredictionInput, type MultiModelResponse } from '../services/api'
import { motion } from 'framer-motion'
import { ResponsiveContainer, BarChart, Bar, Cell, XAxis, YAxis, Tooltip } from 'recharts'

const COLORS = {
  low: '#10B981',
  medium: '#3B82F6',
  high: '#F59E0B',
  veryHigh: '#EF4444',
}

export default function Predict() {
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<MultiModelResponse | null>(null)
  const [error, setError] = useState<string | null>(null)

  const { register, handleSubmit } = useForm<PredictionInput>()

  const getRiskColor = (prob: number) => {
    if (prob >= 0.75) return COLORS.veryHigh
    if (prob >= 0.50) return COLORS.high
    if (prob >= 0.25) return COLORS.medium
    return COLORS.low
  }

  const getAgreementCount = (consensus: MultiModelResponse['consensus']) => {
    if (consensus.consensus_label === 'Churned') {
      return consensus.churn_votes
    }

    return consensus.total_models - consensus.churn_votes
  }

  const onSubmit = async (data: PredictionInput) => {
    setLoading(true)
    setError(null)
    try {
      const response = await apiService.predictCompare(data)
      setResult(response.data)
    } catch (err: any) {
      setError(err.response?.data?.error || 'Prediction failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h2 className="text-2xl font-bold text-white">Customer Risk Predictor</h2>
        <p className="text-muted mt-1">Enter customer profile to get churn probability and multi-model comparison</p>
      </div>

      <div className="grid lg:grid-cols-3 gap-6">
        {/* Input Form */}
        <div className="lg:col-span-1">
          <form onSubmit={handleSubmit(onSubmit)} className="glass-card rounded-xl p-6 space-y-4">
            <h3 className="font-semibold text-white mb-2">Customer Profile</h3>

            {/* Demographics */}
            <div className="space-y-3">
              <p className="text-xs text-muted uppercase tracking-wide">Demographics</p>

              <div>
                <label className="block text-sm text-muted mb-1">Age</label>
                <input
                  type="number"
                  {...register('Customer_Age', { min: 18, max: 100 })}
                  className="w-full bg-card border border-border rounded-lg px-3 py-2 text-white focus:outline-none focus:border-primary"
                  defaultValue={45}
                />
              </div>

              <div>
                <label className="block text-sm text-muted mb-1">Gender</label>
                <select
                  {...register('Gender')}
                  className="w-full bg-card border border-border rounded-lg px-3 py-2 text-white focus:outline-none focus:border-primary"
                  defaultValue="M"
                >
                  <option value="M">Male</option>
                  <option value="F">Female</option>
                </select>
              </div>

              <div>
                <label className="block text-sm text-muted mb-1">Education</label>
                <select
                  {...register('Education_Level')}
                  className="w-full bg-card border border-border rounded-lg px-3 py-2 text-white focus:outline-none focus:border-primary"
                  defaultValue="Graduate"
                >
                  <option>Uneducated</option>
                  <option>High School</option>
                  <option>College</option>
                  <option>Graduate</option>
                  <option>Post-Graduate</option>
                  <option>Doctorate</option>
                  <option>Unknown</option>
                </select>
              </div>

              <div>
                <label className="block text-sm text-muted mb-1">Marital Status</label>
                <select
                  {...register('Marital_Status')}
                  className="w-full bg-card border border-border rounded-lg px-3 py-2 text-white focus:outline-none focus:border-primary"
                  defaultValue="Married"
                >
                  <option>Single</option>
                  <option>Married</option>
                  <option>Divorced</option>
                </select>
              </div>

              <div>
                <label className="block text-sm text-muted mb-1">Income</label>
                <select
                  {...register('Income_Category')}
                  className="w-full bg-card border border-border rounded-lg px-3 py-2 text-white focus:outline-none focus:border-primary"
                  defaultValue="$60K - $80K"
                >
                  <option>Less than $40K</option>
                  <option>$40K - $60K</option>
                  <option>$60K - $80K</option>
                  <option>$80K - $120K</option>
                  <option>$120K +</option>
                  <option>Unknown</option>
                </select>
              </div>
            </div>

            {/* Account Details */}
            <div className="space-y-3 pt-4 border-t border-border">
              <p className="text-xs text-muted uppercase tracking-wide">Account</p>

              <div>
                <label className="block text-sm text-muted mb-1">Card Type</label>
                <select
                  {...register('Card_Category')}
                  className="w-full bg-card border border-border rounded-lg px-3 py-2 text-white focus:outline-none focus:border-primary"
                  defaultValue="Blue"
                >
                  <option>Blue</option>
                  <option>Silver</option>
                  <option>Gold</option>
                  <option>Platinum</option>
                </select>
              </div>

              <div>
                <label className="block text-sm text-muted mb-1">Months on Book</label>
                <input
                  type="number"
                  {...register('Months_on_book', { min: 12, max: 60 })}
                  className="w-full bg-card border border-border rounded-lg px-3 py-2 text-white focus:outline-none focus:border-primary"
                  defaultValue={36}
                />
              </div>

              <div>
                <label className="block text-sm text-muted mb-1">Products Held</label>
                <input
                  type="number"
                  {...register('Total_Relationship_Count', { min: 1, max: 8 })}
                  className="w-full bg-card border border-border rounded-lg px-3 py-2 text-white focus:outline-none focus:border-primary"
                  defaultValue={4}
                />
              </div>

              <div>
                <label className="block text-sm text-muted mb-1">Inactive Months (12mo)</label>
                <input
                  type="number"
                  {...register('Months_Inactive_12_mon', { min: 0, max: 6 })}
                  className="w-full bg-card border border-border rounded-lg px-3 py-2 text-white focus:outline-none focus:border-primary"
                  defaultValue={2}
                />
              </div>

              <div>
                <label className="block text-sm text-muted mb-1">Contacts (12mo)</label>
                <input
                  type="number"
                  {...register('Contacts_Count_12_mon', { min: 0, max: 6 })}
                  className="w-full bg-card border border-border rounded-lg px-3 py-2 text-white focus:outline-none focus:border-primary"
                  defaultValue={3}
                />
              </div>
            </div>

            {/* Transaction Behavior */}
            <div className="space-y-3 pt-4 border-t border-border">
              <p className="text-xs text-muted uppercase tracking-wide">Transactions</p>

              <div>
                <label className="block text-sm text-muted mb-1">Revolving Balance ($)</label>
                <input
                  type="number"
                  {...register('Total_Revolving_Bal', { min: 0, max: 5000 })}
                  className="w-full bg-card border border-border rounded-lg px-3 py-2 text-white focus:outline-none focus:border-primary"
                  defaultValue={1000}
                />
              </div>

              <div>
                <label className="block text-sm text-muted mb-1">Avg Utilization</label>
                <input
                  type="number"
                  step="0.01"
                  min="0"
                  max="1"
                  {...register('Avg_Utilization_Ratio')}
                  className="w-full bg-card border border-border rounded-lg px-3 py-2 text-white focus:outline-none focus:border-primary"
                  defaultValue={0.3}
                />
              </div>

              <div>
                <label className="block text-sm text-muted mb-1">Total Trans Amount ($)</label>
                <input
                  type="number"
                  {...register('Total_Trans_Amt', { min: 0, max: 50000 })}
                  className="w-full bg-card border border-border rounded-lg px-3 py-2 text-white focus:outline-none focus:border-primary"
                  defaultValue={4000}
                />
              </div>

              <div>
                <label className="block text-sm text-muted mb-1">Total Trans Count</label>
                <input
                  type="number"
                  {...register('Total_Trans_Ct', { min: 0, max: 150 })}
                  className="w-full bg-card border border-border rounded-lg px-3 py-2 text-white focus:outline-none focus:border-primary"
                  defaultValue={60}
                />
              </div>

              <div>
                <label className="block text-sm text-muted mb-1">Amt Change Q4→Q1</label>
                <input
                  type="number"
                  step="0.01"
                  min="0"
                  max="5"
                  {...register('Total_Amt_Chng_Q4_Q1')}
                  className="w-full bg-card border border-border rounded-lg px-3 py-2 text-white focus:outline-none focus:border-primary"
                  defaultValue={0.8}
                />
              </div>

              <div>
                <label className="block text-sm text-muted mb-1">Ct Change Q4→Q1</label>
                <input
                  type="number"
                  step="0.01"
                  min="0"
                  max="5"
                  {...register('Total_Ct_Chng_Q4_Q1')}
                  className="w-full bg-card border border-border rounded-lg px-3 py-2 text-white focus:outline-none focus:border-primary"
                  defaultValue={0.7}
                />
              </div>
            </div>

            <button
              type="submit"
              disabled={loading}
              className="w-full bg-primary hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed text-white font-medium py-3 rounded-lg transition-colors"
            >
              {loading ? 'Predicting...' : '⚡ Predict Churn Risk'}
            </button>
          </form>
        </div>

        {/* Results */}
        <div className="lg:col-span-2 space-y-6">
          {error && (
            <div className="bg-danger/20 border border-danger/30 rounded-xl p-4 text-danger">
              {error}
            </div>
          )}

          {!result && !error && (
            <div className="glass-card rounded-xl p-12 text-center">
              <p className="text-6xl mb-4">🔮</p>
              <p className="text-muted">Fill out the form and submit to see predictions</p>
            </div>
          )}

          {result && (
            <>
              {/* Consensus Card */}
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="glass-card rounded-xl p-6"
              >
                <h3 className="text-lg font-semibold text-white mb-4">📈 Model Consensus</h3>
                <div className="grid md:grid-cols-3 gap-4">
                  <div className="text-center p-4 bg-card rounded-lg">
                    <p className="text-sm text-muted mb-2">Consensus</p>
                    <p
                      className="text-3xl font-bold"
                      style={{ color: getRiskColor(result.consensus.average_probability) }}
                    >
                      {result.consensus.consensus_label}
                    </p>
                  </div>
                  <div className="text-center p-4 bg-card rounded-lg">
                    <p className="text-sm text-muted mb-2">Avg Probability</p>
                    <p
                      className="text-3xl font-bold"
                      style={{ color: getRiskColor(result.consensus.average_probability) }}
                    >
                      {result.consensus.average_probability_pct.toFixed(1)}%
                    </p>
                  </div>
                  <div className="text-center p-4 bg-card rounded-lg">
                    <p className="text-sm text-muted mb-2">Model Agreement</p>
                    <p className="text-3xl font-bold text-white">
                      {result.consensus.agreement_pct.toFixed(0)}%
                    </p>
                    <p className="text-xs text-muted mt-1">
                      {getAgreementCount(result.consensus)}/{result.consensus.total_models} models
                    </p>
                  </div>
                </div>
              </motion.div>

              {/* Model Comparison Table */}
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1 }}
                className="glass-card rounded-xl p-6"
              >
                <h3 className="text-lg font-semibold text-white mb-4">Model Predictions</h3>
                <div className="overflow-x-auto">
                  <table className="w-full">
                    <thead>
                      <tr className="border-b border-border">
                        <th className="text-left py-3 px-4 text-muted font-medium">Model</th>
                        <th className="text-right py-3 px-4 text-muted font-medium">Probability</th>
                        <th className="text-right py-3 px-4 text-muted font-medium">Prediction</th>
                        <th className="text-right py-3 px-4 text-muted font-medium">Risk Level</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(result.predictions).map(([key, pred]) => (
                        <tr key={key} className="border-b border-border/50 last:border-0">
                          <td className="py-3 px-4 text-white">{pred.model_name}</td>
                          <td
                            className="text-right py-3 px-4 font-medium"
                            style={{ color: getRiskColor(pred.probability) }}
                          >
                            {(pred.probability * 100).toFixed(1)}%
                          </td>
                          <td className="text-right py-3 px-4">
                            <span
                              className={`px-2 py-1 rounded text-xs font-medium ${
                                pred.prediction_label === 'Churned'
                                  ? 'bg-danger/20 text-danger'
                                  : 'bg-success/20 text-success'
                              }`}
                            >
                              {pred.prediction_label}
                            </span>
                          </td>
                          <td className="text-right py-3 px-4 text-muted">{pred.risk_level}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </motion.div>

              {/* Visualization */}
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2 }}
                className="glass-card rounded-xl p-6"
              >
                <h3 className="text-lg font-semibold text-white mb-4">Probability by Model</h3>
                <ResponsiveContainer width="100%" height={250}>
                  <BarChart data={Object.entries(result.predictions).map(([, pred]) => ({
                    name: pred.model_name.length > 20 ? pred.model_name.substring(0, 20) + '...' : pred.model_name,
                    probability: pred.probability * 100,
                    prediction: pred.prediction_label,
                  }))}>
                    <XAxis tick={{ fill: '#7B82A8', fontSize: 10 }} angle={-45} textAnchor="end" height={80} />
                    <YAxis domain={[0, 100]} tick={{ fill: '#7B82A8' }} />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#12152C', border: '1px solid #1E2440', borderRadius: '8px' }}
                      labelStyle={{ color: '#C8D0E7' }}
                      formatter={(value: number) => [`${value.toFixed(1)}%`, 'Probability']}
                    />
                    <Bar dataKey="probability" radius={[4, 4, 0, 0]}>
                      {Object.entries(result.predictions).map(([, pred], index) => (
                        <Cell
                          key={`cell-${index}`}
                          fill={getRiskColor(pred.probability)}
                        />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </motion.div>
            </>
          )}
        </div>
      </div>
    </div>
  )
}
