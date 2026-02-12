import { useState } from 'react'
import { Box, Paper, Typography, Button, IconButton, Stepper, Step, StepLabel, Stack } from '@mui/material'
import { motion, AnimatePresence } from 'framer-motion'
import { Close, NavigateNext, NavigateBefore, CheckCircle } from '@mui/icons-material'

export default function OnboardingTour({ variant, onComplete, theme }) {
  const [currentStep, setCurrentStep] = useState(0)
  const [isVisible, setIsVisible] = useState(true)

  const chatTourSteps = [
    {
      title: 'Welcome to Kai Chat',
      description: "I'm your AI assistant for PPC marketing tasks. I can help you with audits, creative generation, performance analysis, and strategic insights.",
      highlight: 'header',
    },
    {
      title: 'Ask Me Anything',
      description: "Type your request in the input field below. I'll understand natural language and route your request to the right tool automatically.",
      highlight: 'input',
    },
    {
      title: 'Access Specialized Tools',
      description: 'Use the sidebar to navigate to specialized tools like Klaudit Audit, Creative Studio, PMax Deep Dive, and SERP Monitor.',
      highlight: 'sidebar',
    },
    {
      title: 'Your Chat History',
      description: 'All conversations are automatically saved, so you can pick up where you left off anytime.',
      highlight: 'messages',
    },
  ]

  const auditTourSteps = [
    {
      title: 'Welcome to Klaudit',
      description: 'Automated Google Ads audit generation with 100+ point analysis across 9 dimensions.',
      highlight: 'header',
    },
    {
      title: 'Upload Your Data',
      description: 'Drop CSV or XLSX files from Google Ads exports here. The system supports 14 different file types and can process 50K+ rows.',
      highlight: 'upload',
    },
    {
      title: 'Or Try Demo Data',
      description: "Don't have files ready? Use the demo data option to see how the audit works with sample data.",
      highlight: 'demo',
    },
    {
      title: 'Comprehensive Analysis',
      description: 'The audit takes 5-10 minutes and analyzes structure, keywords, audiences, ads, landing pages, measurement, bidding, cross-channel, and strategy.',
      highlight: 'info',
    },
  ]

  const steps = variant === 'chat' ? chatTourSteps : auditTourSteps

  const handleNext = () => {
    if (currentStep < steps.length - 1) {
      setCurrentStep((prev) => prev + 1)
    } else {
      handleComplete()
    }
  }

  const handleBack = () => {
    setCurrentStep((prev) => Math.max(0, prev - 1))
  }

  const handleSkip = () => {
    setIsVisible(false)
    onComplete?.()
  }

  const handleComplete = () => {
    setIsVisible(false)
    onComplete?.()
  }

  const currentStepData = steps[currentStep]

  if (!isVisible) return null

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          zIndex: 9999,
          pointerEvents: 'none',
        }}
      >
        {/* Backdrop */}
        <Box
          sx={{
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: 'rgba(0, 0, 0, 0.7)',
            backdropFilter: 'blur(4px)',
            pointerEvents: 'auto',
          }}
          onClick={handleSkip}
        />

        {/* Tour Card */}
        <motion.div
          initial={{ scale: 0.9, y: 20 }}
          animate={{ scale: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          style={{
            position: 'absolute',
            top: '50%',
            left: '50%',
            transform: 'translate(-50%, -50%)',
            width: '90%',
            maxWidth: '600px',
            pointerEvents: 'auto',
          }}
        >
          <Paper
            elevation={24}
            sx={{
              p: 4,
              borderRadius: 4,
              background: 'var(--kai-bg)',
              border: `2px solid ${theme.borderColor}`,
              boxShadow: `0 0 60px ${theme.accentColor}40`,
            }}
          >
            {/* Close Button */}
            <Box display="flex" justifyContent="flex-end" mb={2}>
              <IconButton
                size="small"
                onClick={handleSkip}
                sx={{
                  color: '#64748b',
                  '&:hover': { color: 'var(--kai-text)', background: 'var(--kai-surface-muted)' },
                }}
              >
                <Close />
              </IconButton>
            </Box>

            {/* Progress Stepper */}
            <Stepper activeStep={currentStep} sx={{ mb: 4 }}>
              {steps.map((step, index) => (
                <Step key={index}>
                  <StepLabel
                    sx={{
                      '& .MuiStepLabel-label': {
                        color: index === currentStep ? theme.accentColor : '#64748b',
                        fontWeight: index === currentStep ? 600 : 400,
                      },
                      '& .MuiStepIcon-root': {
                        color: index <= currentStep ? theme.accentColor : 'var(--kai-border-strong)',
                      },
                    }}
                  />
                </Step>
              ))}
            </Stepper>

            {/* Step Content */}
            <AnimatePresence mode="wait">
              <motion.div
                key={currentStep}
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20 }}
                transition={{ duration: 0.2 }}
              >
                <Box mb={4}>
                  {/* Icon */}
                  <Box
                    sx={{
                      width: 60,
                      height: 60,
                      borderRadius: '50%',
                      background: `${theme.accentColor}20`,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      mb: 3,
                    }}
                  >
                    {currentStep === steps.length - 1 ? (
                      <CheckCircle sx={{ fontSize: 32, color: theme.accentColor }} />
                    ) : (
                      <Typography variant="h5" fontWeight={700} color={theme.accentColor}>
                        {currentStep + 1}
                      </Typography>
                    )}
                  </Box>

                  {/* Title */}
                  <Typography variant="h5" fontWeight={700} color="var(--kai-text)" gutterBottom>
                    {currentStepData.title}
                  </Typography>

                  {/* Description */}
                  <Typography variant="body1" color="#cbd5e1" sx={{ lineHeight: 1.7 }}>
                    {currentStepData.description}
                  </Typography>
                </Box>
              </motion.div>
            </AnimatePresence>

            {/* Navigation Buttons */}
            <Stack direction="row" spacing={2} justifyContent="space-between" alignItems="center">
              <Button
                onClick={handleSkip}
                sx={{
                  color: '#64748b',
                  textTransform: 'none',
                  '&:hover': { color: 'var(--kai-text)', background: 'var(--kai-surface-muted)' },
                }}
              >
                Skip tour
              </Button>

              <Stack direction="row" spacing={1}>
                {currentStep > 0 && (
                  <Button
                    onClick={handleBack}
                    startIcon={<NavigateBefore />}
                    sx={{
                      color: '#cbd5e1',
                      borderColor: 'var(--kai-border-strong)',
                      textTransform: 'none',
                      '&:hover': { borderColor: theme.borderColor },
                    }}
                    variant="outlined"
                  >
                    Back
                  </Button>
                )}
                <Button
                  onClick={handleNext}
                  endIcon={currentStep < steps.length - 1 ? <NavigateNext /> : null}
                  sx={{
                    background: theme.primaryGradient,
                    textTransform: 'none',
                    px: 3,
                    '&:hover': {
                      background: theme.userBubbleGradient,
                    },
                  }}
                  variant="contained"
                >
                  {currentStep < steps.length - 1 ? 'Next' : 'Get Started'}
                </Button>
              </Stack>
            </Stack>

            {/* Step Indicator */}
            <Typography variant="caption" color="#64748b" textAlign="center" display="block" mt={2}>
              Step {currentStep + 1} of {steps.length}
            </Typography>
          </Paper>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  )
}

