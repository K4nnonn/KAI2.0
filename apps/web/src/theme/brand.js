/**
 * KAI Brand Theme Configuration
 *
 * Central color system based on the KAI (Kelvin AI) logo:
 * - Cool blue gradient (thermometer/cold side)
 * - Warm orange gradient (circuit/hot side)
 * - Temperature spectrum representing AI intelligence
 */

// Primary brand colors
export const brand = {
  // Cool side (Kelvin temperature - cold)
  cool: {
    primary: '#22d3ee',    // Cyan - primary cool
    secondary: '#3b82f6',  // Blue
    tertiary: '#0ea5e9',   // Sky blue
  },

  // Warm side (Kelvin temperature - hot)
  warm: {
    primary: '#f59e0b',    // Amber - primary warm
    secondary: '#ef4444',  // Red/coral
    tertiary: '#fb923c',   // Orange
  },

  // Neutral palette
  neutral: {
    900: 'var(--kai-bg)',  // Darkest background
    800: 'var(--kai-surface-muted)',  // Card backgrounds
    700: 'var(--kai-border-strong)',  // Borders
    600: 'var(--kai-border)',  // Muted text
    500: '#64748b',  // Secondary text
    400: 'var(--kai-text-soft)',  // Body text
    300: '#cbd5e1',  // Light text
    200: 'var(--kai-text)',  // Headings
    100: '#f1f5f9',  // Bright text
  },
}

// Gradient definitions
export const gradients = {
  // Primary brand gradient (blue to orange - temperature spectrum)
  brand: 'linear-gradient(135deg, #22d3ee 0%, #3b82f6 30%, #f59e0b 70%, #ef4444 100%)',
  brandHorizontal: 'linear-gradient(90deg, #22d3ee 0%, #3b82f6 40%, #f59e0b 100%)',
  brandSubtle: 'linear-gradient(135deg, rgba(34, 211, 238, 0.2) 0%, rgba(245, 158, 11, 0.2) 100%)',

  // Cool side gradient
  cool: 'linear-gradient(135deg, #22d3ee, #3b82f6)',
  coolVertical: 'linear-gradient(180deg, #22d3ee, #3b82f6)',

  // Warm side gradient
  warm: 'linear-gradient(135deg, #f59e0b, #ef4444)',
  warmVertical: 'linear-gradient(180deg, #f59e0b, #ef4444)',

  // Sidebar background
  sidebar: 'linear-gradient(180deg, var(--kai-bg) 0%, var(--kai-bg) 100%)',
}

// Tool-specific colors (aligned with brand)
export const toolColors = {
  // Core AI - uses brand gradient feel
  chat: {
    primary: '#22d3ee',     // Cyan (cool side)
    accent: '#0ea5e9',
    gradient: 'linear-gradient(135deg, #22d3ee, #0ea5e9)',
  },

  // Audit - analytical blue
  audit: {
    primary: '#3b82f6',     // Blue
    accent: '#2563eb',
    gradient: 'linear-gradient(135deg, #3b82f6, #2563eb)',
  },

  // PMax - warm/performance
  pmax: {
    primary: '#f472b6',     // Pink (between warm and cool)
    accent: '#ec4899',
    gradient: 'linear-gradient(135deg, #f472b6, #ec4899)',
  },

  // SERP - purple (mix of warm and cool)
  serp: {
    primary: '#a78bfa',     // Purple
    accent: '#8b5cf6',
    gradient: 'linear-gradient(135deg, #a78bfa, #8b5cf6)',
  },

  // Creative - warm/creative energy
  creative: {
    primary: '#f59e0b',     // Amber (warm side)
    accent: '#d97706',
    gradient: 'linear-gradient(135deg, #f59e0b, #d97706)',
  },
}

// Status colors
export const status = {
  success: '#10b981',
  warning: '#f59e0b',
  error: '#ef4444',
  info: '#3b82f6',
}

// Animation tokens
export const animation = {
  fast: '150ms',
  normal: '250ms',
  slow: '400ms',
  easing: 'cubic-bezier(0.4, 0, 0.2, 1)',
}

export default {
  brand,
  gradients,
  toolColors,
  status,
  animation,
}

