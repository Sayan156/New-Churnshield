import { useMemo, useState } from 'react'
import { motion } from 'framer-motion'
import { apiService, type BatchPredictionResponse, type PredictionInput } from '../services/api'

const REQUIRED_COLUMNS: Array<keyof PredictionInput> = [
  'Customer_Age',
  'Gender',
  'Education_Level',
  'Marital_Status',
  'Income_Category',
  'Card_Category',
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
]

type CsvRow = Record<string, string>

function normalizeHeader(header: string) {
  return header.replace(/^\uFEFF/, '').trim()
}

function parseCsvLine(line: string) {
  const values: string[] = []
  let current = ''
  let inQuotes = false

  for (let i = 0; i < line.length; i += 1) {
    const char = line[i]

    if (char === '"') {
      if (inQuotes && line[i + 1] === '"') {
        current += '"'
        i += 1
      } else {
        inQuotes = !inQuotes
      }
      continue
    }

    if (char === ',' && !inQuotes) {
      values.push(current.trim())
      current = ''
      continue
    }

    current += char
  }

  values.push(current.trim())
  return values
}

function parseCsv(text: string) {
  const lines = text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)

  if (lines.length < 2) {
    throw new Error('CSV must include a header row and at least one data row.')
  }

  const headers = parseCsvLine(lines[0]).map((header) => normalizeHeader(header.replace(/^"|"$/g, '')))
  const rows = lines.slice(1).map((line) => {
    const values = parseCsvLine(line)
    return headers.reduce<CsvRow>((acc, header, index) => {
      acc[header] = (values[index] ?? '').replace(/^"|"$/g, '').trim()
      return acc
    }, {})
  })

  return { headers, rows }
}

function toPredictionInput(row: CsvRow): PredictionInput {
  return {
    Customer_Age: Number(row.Customer_Age),
    Gender: row.Gender as PredictionInput['Gender'],
    Education_Level: row.Education_Level,
    Marital_Status: row.Marital_Status,
    Income_Category: row.Income_Category,
    Card_Category: row.Card_Category,
    Months_on_book: Number(row.Months_on_book),
    Total_Relationship_Count: Number(row.Total_Relationship_Count),
    Months_Inactive_12_mon: Number(row.Months_Inactive_12_mon),
    Contacts_Count_12_mon: Number(row.Contacts_Count_12_mon),
    Total_Revolving_Bal: Number(row.Total_Revolving_Bal),
    Total_Amt_Chng_Q4_Q1: Number(row.Total_Amt_Chng_Q4_Q1),
    Total_Trans_Amt: Number(row.Total_Trans_Amt),
    Total_Trans_Ct: Number(row.Total_Trans_Ct),
    Total_Ct_Chng_Q4_Q1: Number(row.Total_Ct_Chng_Q4_Q1),
    Avg_Utilization_Ratio: Number(row.Avg_Utilization_Ratio),
  }
}

function escapeCsvValue(value: string | number) {
  const stringValue = String(value ?? '')
  if (stringValue.includes(',') || stringValue.includes('"') || stringValue.includes('\n')) {
    return `"${stringValue.replace(/"/g, '""')}"`
  }
  return stringValue
}

function triggerCsvDownload(filename: string, headers: string[], rows: Array<Record<string, string | number>>) {
  const csv = [
    headers.map(escapeCsvValue).join(','),
    ...rows.map((row) => headers.map((header) => escapeCsvValue(row[header] ?? '')).join(',')),
  ].join('\n')

  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' })
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  link.click()
  URL.revokeObjectURL(url)
}

export default function BatchPredict() {
  const [fileName, setFileName] = useState('')
  const [rows, setRows] = useState<CsvRow[]>([])
  const [headers, setHeaders] = useState<string[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<BatchPredictionResponse | null>(null)

  const missingColumns = useMemo(
    () => REQUIRED_COLUMNS.filter((column) => !headers.includes(column)),
    [headers],
  )

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    setError(null)
    setResult(null)

    if (!file) {
      return
    }

    try {
      const text = await file.text()
      const parsed = parseCsv(text)
      setRows(parsed.rows)
      setHeaders(parsed.headers)
      setFileName(file.name)
    } catch (uploadError: any) {
      setRows([])
      setHeaders([])
      setFileName('')
      setError(uploadError.message || 'Failed to read CSV file.')
    }
  }

  const handlePredict = async () => {
    if (rows.length === 0) {
      setError('Upload a CSV file before running batch prediction.')
      return
    }

    if (missingColumns.length > 0) {
      setError(`Missing required columns: ${missingColumns.join(', ')}`)
      return
    }

    setLoading(true)
    setError(null)

    try {
      const customers = rows.map(toPredictionInput)
      const response = await apiService.predictBatch(customers)
      const data = response.data as BatchPredictionResponse
      setResult(data)

      const downloadHeaders = [...headers, 'Prediction']
      const downloadRows = rows.map((row, index) => ({
        ...row,
        Prediction: data.predictions[index]?.prediction_label ?? 'Unavailable',
      }))

      const outputName = fileName.replace(/\.csv$/i, '') || 'batch_predictions'
      triggerCsvDownload(`${outputName}_predictions.csv`, downloadHeaders, downloadRows)
    } catch (predictError: any) {
      setError(predictError.response?.data?.error || 'Batch prediction failed.')
    } finally {
      setLoading(false)
    }
  }

  const downloadTemplate = () => {
    triggerCsvDownload(
      'batch_prediction_template.csv',
      REQUIRED_COLUMNS,
      [{
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
      }],
    )
  }

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-white">Batch Prediction</h2>
        <p className="mt-1 text-muted">Upload a CSV, run churn prediction in bulk, and download the same file with a new <span className="text-primary">Prediction</span> column.</p>
      </div>

      <div className="grid gap-6 lg:grid-cols-[1.1fr_0.9fr]">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="glass-card rounded-xl p-6"
        >
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <h3 className="text-lg font-semibold text-white">Upload CSV</h3>
              <p className="text-sm text-muted">Required format: 16 model input columns. Column order does not matter, but header names must match.</p>
            </div>
            <button
              type="button"
              onClick={downloadTemplate}
              className="rounded-lg border border-primary/30 bg-primary/10 px-4 py-2 text-sm font-medium text-primary transition-colors hover:bg-primary/15"
            >
              Download Template
            </button>
          </div>

          <label className="mt-5 flex cursor-pointer flex-col items-center justify-center rounded-xl border border-dashed border-primary/30 bg-card/60 px-6 py-10 text-center transition-colors hover:border-primary/50 hover:bg-card">
            <span className="text-4xl">📄</span>
            <span className="mt-3 text-base font-medium text-white">Choose a CSV file</span>
            <span className="mt-1 text-sm text-muted">One row per customer record</span>
            <input type="file" accept=".csv,text/csv" className="hidden" onChange={handleFileUpload} />
          </label>

          {fileName && (
            <div className="mt-4 rounded-xl border border-border bg-card/60 p-4">
              <p className="text-sm text-muted">Selected file</p>
              <p className="mt-1 font-medium text-white">{fileName}</p>
              <p className="mt-2 text-sm text-muted">{rows.length} rows ready for prediction</p>
            </div>
          )}

          {error && (
            <div className="mt-4 rounded-xl border border-danger/30 bg-danger/10 p-4 text-danger">
              {error}
            </div>
          )}

            <div className="mt-5 flex flex-wrap gap-3">
            <button
              type="button"
              onClick={handlePredict}
              disabled={loading || rows.length === 0}
              className="rounded-lg bg-primary px-5 py-3 font-medium text-background transition-opacity disabled:cursor-not-allowed disabled:opacity-50"
            >
              {loading ? 'Running Predictions...' : 'Run Batch Prediction'}
            </button>
            </div>

          <div className="mt-4 rounded-xl border border-border bg-card/50 p-4 text-sm text-muted">
            The uploader maps values by column name, not column position. You can arrange the required columns in any order.
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.08 }}
          className="glass-card rounded-xl p-6"
        >
          <h3 className="text-lg font-semibold text-white">Required Columns</h3>
          <div className="mt-4 flex flex-wrap gap-2">
            {REQUIRED_COLUMNS.map((column) => (
              <span
                key={column}
                className={`rounded-full px-3 py-1 text-xs ${
                  headers.includes(column)
                    ? 'bg-success/15 text-success'
                    : 'bg-card text-muted'
                }`}
              >
                {column}
              </span>
            ))}
          </div>

          {missingColumns.length > 0 && headers.length > 0 && (
            <div className="mt-4 rounded-xl border border-warning/30 bg-warning/10 p-4 text-sm text-warning">
              Missing columns: {missingColumns.join(', ')}
            </div>
          )}

          {result && (
            <div className="mt-6 space-y-3">
              <div className="rounded-xl border border-border bg-card/60 p-4">
                <p className="text-sm text-muted">Processed rows</p>
                <p className="mt-1 text-2xl font-semibold text-white">{result.count}</p>
              </div>
              <div className="rounded-xl border border-border bg-card/60 p-4">
                <p className="text-sm text-muted">Download</p>
                <p className="mt-1 text-sm text-white">The predicted CSV was downloaded automatically with an added <span className="text-primary">Prediction</span> column.</p>
              </div>
            </div>
          )}
        </motion.div>
      </div>

      {rows.length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.12 }}
          className="glass-card rounded-xl p-6"
        >
          <h3 className="text-lg font-semibold text-white">Preview</h3>
          <div className="mt-4 overflow-x-auto">
            <table className="w-full min-w-[900px]">
              <thead>
                <tr className="border-b border-border">
                  {headers.map((header) => (
                    <th key={header} className="px-3 py-3 text-left text-xs font-medium uppercase tracking-[0.12em] text-muted">
                      {header}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {rows.slice(0, 5).map((row, index) => (
                  <tr key={`${row.Customer_Age}-${index}`} className="border-b border-border/50 last:border-0">
                    {headers.map((header) => (
                      <td key={`${header}-${index}`} className="px-3 py-3 text-sm text-white">
                        {row[header]}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </motion.div>
      )}
    </div>
  )
}
