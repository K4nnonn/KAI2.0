/**
 * KAI Logo Component
 *
 * SVG logo combining:
 * - Thermometer (left) - Kelvin temperature measurement
 * - Circuit nodes (right) - AI/technology
 * - Blue-to-orange gradient - temperature spectrum
 */

export default function KaiLogo({ size = 40, className = '', showText = false }) {
  const height = size
  const width = showText ? size * 2.5 : size

  return (
    <svg
      width={width}
      height={height}
      viewBox={showText ? "0 0 100 40" : "0 0 40 40"}
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={className}
    >
      <defs>
        {/* Main brand gradient - blue to orange */}
        <linearGradient id="kaiGradient" x1="0%" y1="100%" x2="100%" y2="0%">
          <stop offset="0%" stopColor="#22d3ee" />
          <stop offset="35%" stopColor="#3b82f6" />
          <stop offset="65%" stopColor="#f59e0b" />
          <stop offset="100%" stopColor="#ef4444" />
        </linearGradient>

        {/* Cool gradient for thermometer */}
        <linearGradient id="coolGradient" x1="0%" y1="100%" x2="0%" y2="0%">
          <stop offset="0%" stopColor="#22d3ee" />
          <stop offset="100%" stopColor="#3b82f6" />
        </linearGradient>

        {/* Warm gradient for circuits */}
        <linearGradient id="warmGradient" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="#f59e0b" />
          <stop offset="100%" stopColor="#ef4444" />
        </linearGradient>

        {/* Glow effect */}
        <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
          <feGaussianBlur stdDeviation="1" result="blur" />
          <feMerge>
            <feMergeNode in="blur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>

      {/* K shape - Thermometer + Circuit */}
      <g filter="url(#glow)">
        {/* Thermometer vertical bar (left side of K) */}
        <path
          d="M8 6 L8 28 C8 31.3 10.7 34 14 34 C17.3 34 20 31.3 20 28 L20 6 C20 4.9 19.1 4 18 4 L10 4 C8.9 4 8 4.9 8 6Z"
          fill="none"
          stroke="url(#coolGradient)"
          strokeWidth="2.5"
          strokeLinecap="round"
        />

        {/* Thermometer bulb */}
        <circle
          cx="14"
          cy="30"
          r="4"
          fill="url(#coolGradient)"
        />

        {/* Temperature markings */}
        <line x1="12" y1="10" x2="16" y2="10" stroke="url(#coolGradient)" strokeWidth="1.5" strokeLinecap="round" />
        <line x1="12" y1="16" x2="16" y2="16" stroke="url(#coolGradient)" strokeWidth="1.5" strokeLinecap="round" />
        <line x1="12" y1="22" x2="16" y2="22" stroke="url(#coolGradient)" strokeWidth="1.5" strokeLinecap="round" />

        {/* Circuit branches (right side of K) - upper branch */}
        <path
          d="M18 18 L26 10 L32 10"
          fill="none"
          stroke="url(#warmGradient)"
          strokeWidth="2.5"
          strokeLinecap="round"
          strokeLinejoin="round"
        />

        {/* Circuit node - upper */}
        <circle cx="33" cy="10" r="3" fill="url(#warmGradient)" />

        {/* Middle branch */}
        <path
          d="M20 20 L28 20"
          fill="none"
          stroke="url(#kaiGradient)"
          strokeWidth="2"
          strokeLinecap="round"
        />
        <circle cx="30" cy="20" r="2.5" fill="url(#warmGradient)" />

        {/* Circuit branches - lower branch */}
        <path
          d="M18 22 L26 30 L32 30"
          fill="none"
          stroke="url(#warmGradient)"
          strokeWidth="2.5"
          strokeLinecap="round"
          strokeLinejoin="round"
        />

        {/* Circuit node - lower */}
        <circle cx="33" cy="30" r="3" fill="url(#warmGradient)" />

        {/* Small connector nodes */}
        <circle cx="26" cy="10" r="1.5" fill="url(#kaiGradient)" />
        <circle cx="26" cy="30" r="1.5" fill="url(#kaiGradient)" />
      </g>

      {/* Optional text */}
      {showText && (
        <g>
          <text
            x="48"
            y="26"
            fontFamily="system-ui, -apple-system, sans-serif"
            fontSize="18"
            fontWeight="700"
            letterSpacing="-0.5"
          >
            <tspan fill="url(#coolGradient)">K</tspan>
            <tspan fill="url(#kaiGradient)">A</tspan>
            <tspan fill="url(#warmGradient)">I</tspan>
          </text>
          <text
            x="48"
            y="36"
            fontFamily="system-ui, -apple-system, sans-serif"
            fontSize="7"
            fontWeight="500"
            fill="#64748b"
            letterSpacing="0.5"
          >
            kelvin AI
          </text>
        </g>
      )}
    </svg>
  )
}

// Animated version with hover effects
export function KaiLogoAnimated({ size = 40, className = '' }) {
  return (
    <div
      className={className}
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        justifyContent: 'center',
        transition: 'transform 0.2s ease',
        cursor: 'pointer',
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.transform = 'scale(1.05)'
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.transform = 'scale(1)'
      }}
    >
      <KaiLogo size={size} />
    </div>
  )
}
