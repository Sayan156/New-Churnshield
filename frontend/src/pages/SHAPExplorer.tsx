import { useRef, useState, type ChangeEvent } from 'react'
import { useForm } from 'react-hook-form'
import { apiService, type PredictionInput, type SHAPGlobalResponse, type SHAPResponse } from '../services/api'
import { motion } from 'framer-motion'
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, Cell } from 'recharts'

const FEATURE_NAMES: Record<string, string> = {
  Customer_Age: 'Age',
  Gender: 'Gender',
  Education_Level: 'Education',
  Marital_Status: 'Marital Status',
  Income_Category: 'Income',
  Card_Category: 'Card Type',
  Months_on_book: 'Months on Book',
  Total_Relationship_Count: 'Products',
  Months_Inactive_12_mon: 'Inactive Months',
  Contacts_Count_12_mon: 'Contacts',
  Total_Revolving_Bal: 'Revolving Balance',
  Total_Amt_Chng_Q4_Q1: 'Amt Change Q4-Q1',
  Total_Trans_Amt: 'Trans Amount',
  Total_Trans_Ct: 'Trans Count',
  Total_Ct_Chng_Q4_Q1: 'Ct Change Q4-Q1',
  Avg_Utilization_Ratio: 'Utilization',
}

export default function SHAPExplorer() {
  const [activeTab, setActiveTab] = useState<'individual' | 'global'>('individual')
  const [loading, setLoading] = useState(false)
  const [shapResult, setShapResult] = useState<SHAPResponse | null>(null)
  const [globalResult, setGlobalResult] = useState<SHAPGlobalResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [datasetInfo, setDatasetInfo] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement | null>(null)

  const { register, handleSubmit } = useForm<PredictionInput>({
    defaultValues: {
      Customer_Age: 45,
      Gender: 'M',
      Education_Level: 'Graduate',
      Marital_Status: 'Married',
      Income_Category: '$60K - $80K',
      Card_Category: 'Blue',
      Months_on_book: 36,
      Total_Relationship_Count: 4,
      Months_Inactive_12_mon: 2,
      Contacts_Count_12_mon: 3,
      Total_Revolving_Bal: 1000,
      Total_Amt_Chng_Q4_Q1: 0.8,
      Total_Trans_Amt: 4000,
      Total_Trans_Ct: 60,
      Total_Ct_Chng_Q4_Q1: 0.7,
      Avg_Utilization_Ratio: 0.3,
    },
  })

  const onSubmit = async (data: PredictionInput) => {
    if (activeTab !== 'individual') return

    setLoading(true)
    setError(null)
    setGlobalResult(null)
    try {
      const response = await apiService.getShapIndividual(data)
      setShapResult(response.data)
    } catch (err: any) {
      setError(err.response?.data?.error || 'SHAP computation failed')
    } finally {
      setLoading(false)
    }
  }

  const loadSampleGlobalAnalysis = async () => {
    setLoading(true)
    setError(null)
    setShapResult(null)
    setDatasetInfo('Using backend sample dataset')

    try {
      const response = await apiService.getShapGlobal([], 50)
      setGlobalResult(response.data)
    } catch (err: any) {
      setError(err.response?.data?.error || 'Global SHAP computation failed')
    } finally {
      setLoading(false)
    }
  }

  const handleUploadClick = () => {
    fileInputRef.current?.click()
  }

  const parseCsvValue = (header: string, value: string) => {
    const numericFields = new Set([
      'Customer_Age',
      'Months_on_book',
      'Total_Relationship_Count',
      'Months_Inactive_12_mon',
      'Contacts_Count_12_mon',
      'Total_Revolving_Bal',
      'Total_Amt_Chng_Q4_Q1',
      'Total_Trans_Amt',
      'Total_Trans_Ct',
      'Total_Ct_Chng_Q4_Q1',
      'Avg_Utilization_Ratio',
    ])

    if (!numericFields.has(header)) return value

    const parsed = Number(value)
    if (Number.isNaN(parsed)) {
      throw new Error(`Invalid numeric value "${value}" for column ${header}`)
    }
    return parsed
  }

  const parseCsv = (text: string): PredictionInput[] => {
    const lines = text
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter(Boolean)

    if (lines.length < 2) {
      throw new Error('CSV file must include a header row and at least one data row')
    }

    const headers = lines[0].split(',').map((header) => header.trim())
    const requiredHeaders = Object.keys(FEATURE_NAMES)
    const missingHeaders = requiredHeaders.filter((header) => !headers.includes(header))

    if (missingHeaders.length > 0) {
      throw new Error(`Missing required columns: ${missingHeaders.join(', ')}`)
    }

    return lines.slice(1).map((line, index) => {
      const values = line.split(',').map((value) => value.trim())
      if (values.length !== headers.length) {
        throw new Error(`CSV row ${index + 2} does not match the header column count`)
      }

      const record = {} as Record<string, string | number>
      headers.forEach((header, columnIndex) => {
        record[header] = parseCsvValue(header, values[columnIndex])
      })

      return record as unknown as PredictionInput
    })
  }

  const handleFileSelected = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return

    setLoading(true)
    setError(null)
    setShapResult(null)

    try {
      const csvText = await file.text()
      const records = parseCsv(csvText)
      setDatasetInfo(`Uploaded ${file.name} (${records.length} rows)`)

      const response = await apiService.getShapGlobal(records, Math.min(records.length, 100))
      setGlobalResult(response.data)
    } catch (err: any) {
      setGlobalResult(null)
      setError(err.response?.data?.error || err.message || 'Failed to process CSV dataset')
    } finally {
      event.target.value = ''
      setLoading(false)
    }
  }

  const getRiskColor = (prob: number) => {
    if (prob >= 0.75) return '#EF4444'
    if (prob >= 0.50) return '#F59E0B'
    if (prob >= 0.25) return '#3B82F6'
    return '#10B981'
  }

  const getRiskLabel = (prob: number) => {
    if (prob >= 0.75) return 'Very High Risk'
    if (prob >= 0.50) return 'High Risk'
    if (prob >= 0.25) return 'Medium Risk'
    return 'Low Risk'
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h2 className="text-2xl font-bold text-white">SHAP Explorer</h2>
        <p className="text-muted mt-1">Understand why the model predicts churn</p>
      </div>

      {/* Tabs */}
      <div className="flex gap-2 border-b border-border">
        <button
          onClick={() => {
            setActiveTab('individual')
            setError(null)
          }}
          className={`px-4 py-2 font-medium transition-colors ${
            activeTab === 'individual'
              ? 'text-primary border-b-2 border-primary'
              : 'text-muted hover:text-white'
          }`}
        >
          👤 Individual Explanation
        </button>
        <button
          onClick={() => {
            setActiveTab('global')
            setError(null)
          }}
          className={`px-4 py-2 font-medium transition-colors ${
            activeTab === 'global'
              ? 'text-primary border-b-2 border-primary'
              : 'text-muted hover:text-white'
          }`}
        >
          🌐 Global Analysis
        </button>
      </div>

      {/* Individual Tab */}
      {activeTab === 'individual' && (
        <div className="grid lg:grid-cols-3 gap-6">
          {/* Input Form */}
          <div className="lg:col-span-1">
            <form onSubmit={handleSubmit(onSubmit)} className="glass-card rounded-xl p-6 space-y-4">
              <h3 className="font-semibold text-white mb-2">Customer Profile</h3>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-sm text-muted mb-1">Age</label>
                  <input
                    type="number"
                    {...register('Customer_Age')}
                    className="w-full bg-card border border-border rounded-lg px-3 py-2 text-white focus:outline-none focus:border-primary"
                  />
                </div>
                <div>
                  <label className="block text-sm text-muted mb-1">Gender</label>
                  <select {...register('Gender')} className="w-full bg-card border border-border rounded-lg px-3 py-2 text-white focus:outline-none focus:border-primary">
                    <option value="M">M</option>
                    <option value="F">F</option>
                  </select>
                </div>
              </div>

              <div>
                <label className="block text-sm text-muted mb-1">Education</label>
                <select {...register('Education_Level')} className="w-full bg-card border border-border rounded-lg px-3 py-2 text-white focus:outline-none focus:border-primary">
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
                <label className="block text-sm text-muted mb-1">Income</label>
                <select {...register('Income_Category')} className="w-full bg-card border border-border rounded-lg px-3 py-2 text-white focus:outline-none focus:border-primary">
                  <option>Less than $40K</option>
                  <option>$40K - $60K</option>
                  <option>$60K - $80K</option>
                  <option>$80K - $120K</option>
                  <option>$120K +</option>
                  <option>Unknown</option>
                </select>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-sm text-muted mb-1">Trans Amount</label>
                  <input type="number" {...register('Total_Trans_Amt')} className="w-full bg-card border border-border rounded-lg px-3 py-2 text-white focus:outline-none focus:border-primary" />
                </div>
                <div>
                  <label className="block text-sm text-muted mb-1">Trans Count</label>
                  <input type="number" {...register('Total_Trans_Ct')} className="w-full bg-card border border-border rounded-lg px-3 py-2 text-white focus:outline-none focus:border-primary" />
                </div>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-sm text-muted mb-1">Inactive Months</label>
                  <input type="number" {...register('Months_Inactive_12_mon')} className="w-full bg-card border border-border rounded-lg px-3 py-2 text-white focus:outline-none focus:border-primary" />
                </div>
                <div>
                  <label className="block text-sm text-muted mb-1">Contacts</label>
                  <input type="number" {...register('Contacts_Count_12_mon')} className="w-full bg-card border border-border rounded-lg px-3 py-2 text-white focus:outline-none focus:border-primary" />
                </div>
              </div>

              <div>
                <label className="block text-sm text-muted mb-1">Utilization Ratio</label>
                <input type="number" step="0.01" min="0" max="1" {...register('Avg_Utilization_Ratio')} className="w-full bg-card border border-border rounded-lg px-3 py-2 text-white focus:outline-none focus:border-primary" />
              </div>

              <button
                type="submit"
                disabled={loading}
                className="w-full bg-primary hover:bg-primary/90 disabled:opacity-50 text-white font-medium py-3 rounded-lg transition-colors"
              >
                {loading ? 'Computing SHAP...' : '🔍 Explain Prediction'}
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

            {!shapResult && !error && (
              <div className="glass-card rounded-xl p-12 text-center">
                <p className="text-6xl mb-4">🔍</p>
                <p className="text-muted">Submit a customer profile to see SHAP explanation</p>
                <p className="text-sm text-muted mt-2">
                  Note: SHAP computation may take 15-30 seconds
                </p>
              </div>
            )}

            {shapResult && (
              <>
                {/* Prediction Summary */}
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="glass-card rounded-xl p-6"
                >
                  <h3 className="text-lg font-semibold text-white mb-4">Prediction Summary</h3>
                  <div className="grid md:grid-cols-3 gap-4">
                    <div className="text-center p-4 bg-card rounded-lg">
                      <p className="text-sm text-muted mb-2">Probability</p>
                      <p
                        className="text-2xl font-bold"
                        style={{ color: getRiskColor(shapResult.probability) }}
                      >
                        {(shapResult.probability * 100).toFixed(1)}%
                      </p>
                    </div>
                    <div className="text-center p-4 bg-card rounded-lg">
                      <p className="text-sm text-muted mb-2">Prediction</p>
                      <p className="text-2xl font-bold text-white">
                        {shapResult.prediction === 1 ? 'Churned' : 'Retained'}
                      </p>
                    </div>
                    <div className="text-center p-4 bg-card rounded-lg">
                      <p className="text-sm text-muted mb-2">Risk Level</p>
                      <p
                        className="text-lg font-bold"
                        style={{ color: getRiskColor(shapResult.probability) }}
                      >
                        {getRiskLabel(shapResult.probability)}
                      </p>
                    </div>
                  </div>
                </motion.div>

                {/* Feature Importance */}
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.1 }}
                  className="glass-card rounded-xl p-6"
                >
                  <h3 className="text-lg font-semibold text-white mb-4">Top Features Impacting Prediction</h3>
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart
                      data={shapResult.importance_ranking.slice(0, 10).map((item) => ({
                        feature: FEATURE_NAMES[item.feature] || item.feature,
                        impact: Math.abs(item.shap_value),
                        shapValue: item.shap_value,
                      }))}
                      layout="vertical"
                      margin={{ left: 80 }}
                    >
                      <XAxis type="number" tick={{ fill: '#7B82A8' }} />
                      <YAxis dataKey="feature" type="category" tick={{ fill: '#C8D0E7', fontSize: 12 }} />
                      <Tooltip
                        contentStyle={{ backgroundColor: '#12152C', border: '1px solid #1E2440', borderRadius: '8px' }}
                        labelStyle={{ color: '#C8D0E7' }}
                        formatter={(value: number, name: string) => {
                          if (name === 'impact') {
                            return [value.toFixed(4), 'Impact']
                          }
                          return [value.toFixed(4), 'SHAP Value']
                        }}
                      />
                      <Bar dataKey="impact" radius={[0, 4, 4, 0]}>
                        {shapResult.importance_ranking.slice(0, 10).map((item, index) => (
                          <Cell
                            key={`cell-${index}`}
                            fill={item.shap_value > 0 ? '#EF4444' : '#10B981'}
                          />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                  <p className="text-xs text-muted mt-4">
                    <span className="text-danger">■</span> Red: Increases churn probability
                    <span className="text-success ml-4">■</span> Green: Decreases churn probability
                  </p>
                </motion.div>

                {/* Detailed SHAP Values */}
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.2 }}
                  className="glass-card rounded-xl p-6"
                >
                  <h3 className="text-lg font-semibold text-white mb-4">Detailed Feature Contributions</h3>
                  <div className="space-y-3">
                    {shapResult.top_features.map((feature) => (
                      <div key={feature.feature} className="flex items-center gap-4">
                        <span className="text-sm text-muted w-32 truncate">{FEATURE_NAMES[feature.feature] || feature.feature}</span>
                        <div className="flex-1 h-6 bg-card rounded-full overflow-hidden relative">
                          <div
                            className="absolute top-0 bottom-0 w-0.5 bg-white/30"
                            style={{ left: '50%' }}
                          />
                          <div
                            className={`absolute top-0 bottom-0 ${feature.shap_value > 0 ? 'right-1/2' : 'left-1/2'}`}
                            style={{
                              width: `${Math.min(Math.abs(feature.shap_value) * 100, 50)}%`,
                              backgroundColor: feature.shap_value > 0 ? '#EF4444' : '#10B981',
                            }}
                          />
                        </div>
                        <span
                          className={`text-sm font-medium w-20 text-right ${
                            feature.shap_value > 0 ? 'text-danger' : 'text-success'
                          }`}
                        >
                          {feature.shap_value > 0 ? '+' : ''}{feature.shap_value.toFixed(4)}
                        </span>
                      </div>
                    ))}
                  </div>
                </motion.div>
              </>
            )}
          </div>
        </div>
      )}

      {/* Global Tab */}
      {activeTab === 'global' && (
        <div className="space-y-6">
          <input
            ref={fileInputRef}
            type="file"
            accept=".csv,text/csv"
            className="hidden"
            onChange={handleFileSelected}
          />

          <div className="glass-card rounded-xl p-8 text-center">
            <p className="text-6xl mb-4">🌐</p>
            <h3 className="text-xl font-semibold text-white mb-2">Global SHAP Analysis</h3>
            <p className="text-muted mb-6">
              Upload a CSV with the model input columns or use the backend sample dataset
            </p>
            <div className="flex flex-col sm:flex-row justify-center gap-4">
              <button
                onClick={handleUploadClick}
                disabled={loading}
                className="px-6 py-3 bg-primary hover:bg-primary/90 disabled:opacity-50 text-white font-medium rounded-lg transition-colors"
              >
                {loading ? 'Processing...' : '📁 Upload Dataset'}
              </button>
              <button
                onClick={loadSampleGlobalAnalysis}
                disabled={loading}
                className="px-6 py-3 bg-card hover:bg-card/80 disabled:opacity-50 text-white font-medium rounded-lg border border-border transition-colors"
              >
                {loading ? 'Computing...' : '📊 Use Sample Data'}
              </button>
            </div>
            <p className="text-sm text-muted mt-6">
              Note: Global SHAP computation requires processing multiple samples and may take 1-2 minutes
            </p>
            {datasetInfo && (
              <p className="text-sm text-primary mt-3">{datasetInfo}</p>
            )}
          </div>

          {error && (
            <div className="bg-danger/20 border border-danger/30 rounded-xl p-4 text-danger">
              {error}
            </div>
          )}

          {globalResult && (
            <>
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="glass-card rounded-xl p-6"
              >
                <h3 className="text-lg font-semibold text-white mb-4">Global Summary</h3>
                <div className="grid md:grid-cols-3 gap-4">
                  <div className="text-center p-4 bg-card rounded-lg">
                    <p className="text-sm text-muted mb-2">Rows Explained</p>
                    <p className="text-2xl font-bold text-white">{globalResult.n_samples}</p>
                  </div>
                  <div className="text-center p-4 bg-card rounded-lg">
                    <p className="text-sm text-muted mb-2">Feature Count</p>
                    <p className="text-2xl font-bold text-white">{globalResult.feature_names.length}</p>
                  </div>
                  <div className="text-center p-4 bg-card rounded-lg">
                    <p className="text-sm text-muted mb-2">SHAP Matrix</p>
                    <p className="text-2xl font-bold text-white">
                      {globalResult.shap_values_shape.join(' × ')}
                    </p>
                  </div>
                </div>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1 }}
                className="glass-card rounded-xl p-6"
              >
                <h3 className="text-lg font-semibold text-white mb-4">Global Feature Importance</h3>
                <ResponsiveContainer width="100%" height={380}>
                  <BarChart
                    data={globalResult.feature_importance.slice(0, 10).map((item) => ({
                      feature: FEATURE_NAMES[item.feature] || item.feature,
                      importance: item.importance,
                    }))}
                    layout="vertical"
                    margin={{ left: 80 }}
                  >
                    <XAxis type="number" tick={{ fill: '#7B82A8' }} />
                    <YAxis dataKey="feature" type="category" tick={{ fill: '#C8D0E7', fontSize: 12 }} />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#12152C', border: '1px solid #1E2440', borderRadius: '8px' }}
                      labelStyle={{ color: '#C8D0E7' }}
                      formatter={(value: number) => [value.toFixed(4), 'Mean |SHAP|']}
                    />
                    <Bar dataKey="importance" radius={[0, 4, 4, 0]} fill="#3B82F6">
                      {globalResult.feature_importance.slice(0, 10).map((_, index) => (
                        <Cell key={`global-cell-${index}`} fill={index < 3 ? '#EF4444' : '#3B82F6'} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </motion.div>
            </>
          )}
        </div>
      )}
    </div>
  )
}
