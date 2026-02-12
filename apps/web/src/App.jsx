import { Routes, Route } from 'react-router-dom'
import KaiChat from './pages/KaiChat'
import KlauditAudit from './pages/KlauditAudit'
import CreativeStudio from './pages/CreativeStudio'
import PMaxDeepDive from './pages/PMaxDeepDive'
import SerpMonitor from './pages/SerpMonitor'
import Settings from './pages/Settings'
import EnvManager from './pages/EnvManager'
import Sa360Columns from './pages/Sa360Columns'
import ArchitectureInfo from './pages/ArchitectureInfo'
import ArchitectureInfoShare from './pages/ArchitectureInfoShare'
import DashboardTest from './components/dashboards/__tests__/DashboardTest'
import Shell from './components/Shell'
import PasswordGate from './components/PasswordGate'

function App() {
  return (
    <PasswordGate>
      <Routes>
        <Route
        path="/"
        element={
          <Shell>
            <KaiChat />
          </Shell>
        }
      />
      <Route
        path="/klaudit"
        element={
          <Shell>
            <KlauditAudit
              variant="audit"
              title="Klaudit Audit"
              subtitle="Upload exports or describe the account. I'll generate a clean, AI-enhanced audit."
            />
          </Shell>
        }
      />
      <Route
        path="/creative-studio"
        element={
          <Shell>
            <CreativeStudio />
          </Shell>
        }
      />
      <Route
        path="/pmax"
        element={
          <Shell>
            <PMaxDeepDive />
          </Shell>
        }
      />
      <Route
        path="/serp"
        element={
          <Shell>
            <SerpMonitor />
          </Shell>
        }
      />
      <Route
        path="/settings"
        element={
          <Shell>
            <Settings />
          </Shell>
        }
      />
      <Route
        path="/env"
        element={
          <Shell>
            <EnvManager />
          </Shell>
        }
      />
      <Route
        path="/sa360-columns"
        element={
          <Shell>
            <Sa360Columns />
          </Shell>
        }
      />
      <Route
        path="/info"
        element={
          <Shell>
            <ArchitectureInfo />
          </Shell>
        }
      />
      <Route
        path="/info-share"
        element={<ArchitectureInfoShare />}
      />
      <Route
        path="/dashboard-test"
        element={
          <Shell>
            <DashboardTest />
          </Shell>
        }
      />
      </Routes>
    </PasswordGate>
  )
}

export default App
