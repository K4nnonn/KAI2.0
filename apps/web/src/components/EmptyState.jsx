import { Box, Paper, Typography, Chip, Stack, Accordion, AccordionSummary, AccordionDetails, Button } from '@mui/material'
import { motion } from 'framer-motion'
import {
  Chat,
  Assessment,
  Brush,
  TrendingUp,
  Search,
  Lightbulb,
  CloudUpload,
  Science,
  ExpandMore,
  InfoOutlined
} from '@mui/icons-material'

export default function EmptyState({ variant, onSuggestedPrompt, theme }) {
  const chatContent = {
    icon: <Chat sx={{ fontSize: 48, color: theme.accentColor }} />,
    title: "Hi! I'm Kai, your PPC marketing assistant",
    description: "I can help you with comprehensive audits, creative generation, performance analysis, and strategic insights.",
    capabilities: [
      { icon: <Assessment />, label: 'Generate comprehensive audits', color: theme.accentColor },
      { icon: <Brush />, label: 'Create RSA headlines', color: '#ec4899' },
      { icon: <TrendingUp />, label: 'Analyze PMax performance', color: '#f59e0b' },
      { icon: <Search />, label: 'Check URL health & SERP', color: '#10b981' },
      { icon: <Lightbulb />, label: 'Answer PPC strategy questions', color: '#8b5cf6' },
    ],
    suggestedPrompts: [
      "Generate a Brand audit with demo data",
      "Create headlines for my SaaS company",
      "Why is my PMax campaign underperforming?",
      "Check URL health for my landing pages",
      "What's the benchmark CTR for Financial Services?",
    ],
    footer: "ðŸ’¬ Chat history is saved automatically",
  }

  const auditContent = {
    icon: <Assessment sx={{ fontSize: 48, color: theme.accentColor }} />,
    title: "Klaudit: 100+ Point Google Ads Audit",
    description: "Comprehensive analysis of your Google Ads campaigns with industry benchmarks and actionable recommendations.",
    features: [
      '9 audit dimensions analyzed',
      'Industry benchmark comparisons',
      'Business impact scoring (1-5 scale)',
      'Detailed recommendations with evidence',
      'Color-coded Excel report',
    ],
    specs: [
      { label: 'Analysis time', value: '5-10 minutes', icon: 'â±ï¸' },
      { label: 'Accepts', value: 'CSV, XLSX (up to 100MB)', icon: 'ðŸ“' },
      { label: 'Row capacity', value: '50K+ rows supported', icon: 'ðŸ“Š' },
    ],
    fileTypes: [
      { name: 'account.csv', description: 'Account metadata', required: false },
      { name: 'campaign_details.csv', description: 'Campaign structure', required: true },
      { name: 'ad_group_details.csv', description: 'Ad group data', required: true },
      { name: 'search_keyword.csv', description: 'Keyword performance (60K+ rows supported)', required: true },
      { name: 'ad.csv', description: 'Headlines, descriptions', required: false },
      { name: 'landing_page.csv', description: 'Landing page analysis', required: false },
      { name: 'audience_segment.csv', description: 'Audience setup', required: false },
      { name: 'ad_schedule.csv', description: 'Day/hour targeting', required: false },
      { name: 'change_history.csv', description: 'Recent modifications', required: false },
      { name: 'callout_extension.csv', description: 'Callout assets', required: false },
      { name: 'sitelink_extension.csv', description: 'Sitelink assets', required: false },
    ],
  }

  const content = variant === 'chat' ? chatContent : auditContent

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <Paper
        elevation={0}
        sx={{
          p: 4,
          borderRadius: 3,
          border: `2px dashed ${theme.borderColor}33`,
          background: `${theme.headerBg}80`,
          textAlign: 'center',
        }}
      >
        <Box display="flex" flexDirection="column" alignItems="center" gap={3}>
          {/* Icon */}
          <Box
            sx={{
              width: 80,
              height: 80,
              borderRadius: '50%',
              background: `${theme.accentColor}15`,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            {content.icon}
          </Box>

          {/* Title */}
          <Typography variant="h5" fontWeight={700} color="var(--kai-text)">
            {content.title}
          </Typography>

          {/* Description */}
          <Typography variant="body1" color="#cbd5e1" maxWidth="600px">
            {content.description}
          </Typography>

          {variant === 'chat' ? (
            <>
              {/* Capabilities Grid */}
              <Box sx={{ width: '100%', maxWidth: '600px' }}>
                <Typography variant="subtitle2" color="var(--kai-text-soft)" mb={2} textAlign="left">
                  I can help you with:
                </Typography>
                <Stack spacing={1.5}>
                  {content.capabilities.map((cap, idx) => (
                    <Box
                      key={idx}
                      sx={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: 2,
                        p: 1.5,
                        borderRadius: 2,
                        background: 'var(--kai-bg)',
                        border: '1px solid var(--kai-surface-alt)',
                      }}
                    >
                      <Box sx={{ color: cap.color }}>{cap.icon}</Box>
                      <Typography variant="body2" color="var(--kai-text)">
                        {cap.label}
                      </Typography>
                    </Box>
                  ))}
                </Stack>
              </Box>

              {/* Suggested Prompts */}
              <Box sx={{ width: '100%', maxWidth: '600px' }}>
                <Typography variant="subtitle2" color="var(--kai-text-soft)" mb={2} textAlign="left">
                  Try asking:
                </Typography>
                <Stack spacing={1}>
                  {content.suggestedPrompts.map((prompt, idx) => (
                    <Chip
                      key={idx}
                      label={prompt}
                      onClick={() => onSuggestedPrompt && onSuggestedPrompt(prompt)}
                      sx={{
                        justifyContent: 'flex-start',
                        height: 'auto',
                        py: 1,
                        px: 2,
                        background: theme.aiBubbleBg,
                        border: `1px solid ${theme.borderColor}33`,
                        color: 'var(--kai-text)',
                        '&:hover': {
                          background: `${theme.accentColor}20`,
                          borderColor: theme.accentColor,
                        },
                        '& .MuiChip-label': {
                          whiteSpace: 'normal',
                          textAlign: 'left',
                        },
                      }}
                    />
                  ))}
                </Stack>
              </Box>

              {/* Footer */}
              <Typography variant="caption" color="#64748b">
                {content.footer}
              </Typography>
            </>
          ) : (
            <>
              {/* Features List */}
              <Box sx={{ width: '100%', maxWidth: '600px' }}>
                <Typography variant="subtitle2" color="var(--kai-text-soft)" mb={2} textAlign="left">
                  What you'll get:
                </Typography>
                <Stack spacing={1}>
                  {content.features.map((feature, idx) => (
                    <Box key={idx} display="flex" alignItems="center" gap={1.5}>
                      <Box
                        sx={{
                          width: 6,
                          height: 6,
                          borderRadius: '50%',
                          background: theme.accentColor,
                        }}
                      />
                      <Typography variant="body2" color="var(--kai-text)">
                        {feature}
                      </Typography>
                    </Box>
                  ))}
                </Stack>
              </Box>

              {/* Specs */}
              <Box sx={{ width: '100%', maxWidth: '600px' }}>
                <Stack direction="row" spacing={2} justifyContent="center" flexWrap="wrap">
                  {content.specs.map((spec, idx) => (
                    <Paper
                      key={idx}
                      elevation={0}
                      sx={{
                        p: 2,
                        borderRadius: 2,
                        background: 'var(--kai-bg)',
                        border: '1px solid var(--kai-surface-alt)',
                        minWidth: '180px',
                      }}
                    >
                      <Typography variant="caption" color="#64748b">
                        {spec.icon} {spec.label}
                      </Typography>
                      <Typography variant="body2" color="var(--kai-text)" fontWeight={600} mt={0.5}>
                        {spec.value}
                      </Typography>
                    </Paper>
                  ))}
                </Stack>
              </Box>

              {/* File Types Accordion */}
              <Box sx={{ width: '100%', maxWidth: '600px' }}>
                <Accordion
                  sx={{
                    background: 'var(--kai-bg)',
                    border: '1px solid var(--kai-surface-alt)',
                    '&:before': { display: 'none' },
                  }}
                >
                  <AccordionSummary
                    expandIcon={<ExpandMore sx={{ color: theme.accentColor }} />}
                    sx={{ '& .MuiAccordionSummary-content': { alignItems: 'center', gap: 1 } }}
                  >
                    <InfoOutlined sx={{ fontSize: 18, color: theme.accentColor }} />
                    <Typography variant="body2" color="var(--kai-text)">
                      Supported file types (14 total)
                    </Typography>
                  </AccordionSummary>
                  <AccordionDetails>
                    <Stack spacing={1}>
                      <Typography variant="caption" color="var(--kai-text-soft)" fontWeight={600} mb={1}>
                        Required for comprehensive audit:
                      </Typography>
                      {content.fileTypes
                        .filter((f) => f.required)
                        .map((file, idx) => (
                          <Box key={idx} display="flex" gap={1.5} alignItems="start">
                            <Box sx={{ color: theme.accentColor, mt: 0.5 }}>âœ“</Box>
                            <Box>
                              <Typography variant="body2" color="var(--kai-text)" fontWeight={600}>
                                {file.name}
                              </Typography>
                              <Typography variant="caption" color="#64748b">
                                {file.description}
                              </Typography>
                            </Box>
                          </Box>
                        ))}

                      <Typography variant="caption" color="var(--kai-text-soft)" fontWeight={600} mt={2} mb={1}>
                        Optional (enhances analysis):
                      </Typography>
                      {content.fileTypes
                        .filter((f) => !f.required)
                        .map((file, idx) => (
                          <Box key={idx} display="flex" gap={1.5} alignItems="start">
                            <Box sx={{ color: '#64748b', mt: 0.5 }}>â—‹</Box>
                            <Box>
                              <Typography variant="body2" color="#cbd5e1">
                                {file.name}
                              </Typography>
                              <Typography variant="caption" color="#64748b">
                                {file.description}
                              </Typography>
                            </Box>
                          </Box>
                        ))}

                      <Typography variant="caption" color="#64748b" mt={2}>
                        âš¡ Validation is non-blocking (works with partial data)
                      </Typography>
                    </Stack>
                  </AccordionDetails>
                </Accordion>
              </Box>

              {/* CTA Buttons */}
              <Stack direction="row" spacing={2} mt={2}>
                <Button
                  variant="contained"
                  startIcon={<CloudUpload />}
                  sx={{
                    background: theme.primaryGradient,
                    textTransform: 'none',
                    px: 3,
                    '&:hover': {
                      background: theme.userBubbleGradient,
                    },
                  }}
                  onClick={() => document.getElementById('file-upload-input')?.click()}
                >
                  Upload Files
                </Button>
                <Button
                  variant="outlined"
                  startIcon={<Science />}
                  sx={{
                    borderColor: theme.borderColor,
                    color: theme.accentColor,
                    textTransform: 'none',
                    px: 3,
                    '&:hover': {
                      borderColor: theme.accentColor,
                      background: `${theme.accentColor}10`,
                    },
                  }}
                  onClick={() => onSuggestedPrompt && onSuggestedPrompt('Generate a Brand audit with demo data')}
                >
                  Try Demo Data
                </Button>
              </Stack>
            </>
          )}
        </Box>
      </Paper>
    </motion.div>
  )
}

