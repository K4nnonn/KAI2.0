import { useLocation } from 'react-router-dom'
import Sidebar from './Sidebar'
import './Shell.css'

import DashboardIcon from '@mui/icons-material/Dashboard'
import ChatIcon from '@mui/icons-material/Chat'
import BrushIcon from '@mui/icons-material/Brush'
import TimelineIcon from '@mui/icons-material/Timeline'
import SearchIcon from '@mui/icons-material/Search'
import SettingsIcon from '@mui/icons-material/Settings'
import KeyIcon from '@mui/icons-material/VpnKey'
import InfoOutlinedIcon from '@mui/icons-material/InfoOutlined'
import ViewColumnIcon from '@mui/icons-material/ViewColumn'

export default function Shell({ children }) {
  const location = useLocation()

  const navItems = [
    { label: 'Kai Chat', icon: <ChatIcon fontSize="small" />, path: '/' },
    { label: 'Klaudit Audit', icon: <DashboardIcon fontSize="small" />, path: '/klaudit' },
    { label: 'Creative Studio', icon: <BrushIcon fontSize="small" />, path: '/creative-studio' },
    { label: 'PMax Deep Dive', icon: <TimelineIcon fontSize="small" />, path: '/pmax' },
    { label: 'SERP Monitor', icon: <SearchIcon fontSize="small" />, path: '/serp' },
    { label: 'Env & Keys', icon: <KeyIcon fontSize="small" />, path: '/env' },
    { label: 'SA360 Columns', icon: <ViewColumnIcon fontSize="small" />, path: '/sa360-columns' },
    { label: 'Info', icon: <InfoOutlinedIcon fontSize="small" />, path: '/info' },
    { label: 'Settings', icon: <SettingsIcon fontSize="small" />, path: '/settings' },
  ]

  return (
    <div className="shell">
      <Sidebar items={navItems} activePath={location.pathname} />
      <main className="shell-main">{children}</main>
    </div>
  )
}
