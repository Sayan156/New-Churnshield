import { Routes, Route, Navigate } from 'react-router-dom'
import Layout from './components/Layout'
import Dashboard from './pages/Dashboard'
import Predict from './pages/Predict'
import BatchPredict from './pages/BatchPredict'
import Arena from './pages/Arena'
import SHAPExplorer from './pages/SHAPExplorer'

function App() {
  return (
    <Routes>
      <Route path="/" element={<Layout />}>
        <Route index element={<Navigate to="/dashboard" replace />} />
        <Route path="dashboard" element={<Dashboard />} />
        <Route path="predict" element={<Predict />} />
        <Route path="batch" element={<BatchPredict />} />
        <Route path="arena" element={<Arena />} />
        <Route path="shap" element={<SHAPExplorer />} />
      </Route>
    </Routes>
  )
}

export default App
