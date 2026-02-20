# Microphone Test Guide

## Quick Test

Record 5 seconds and play back:

```bash
# Record
ffmpeg -f alsa -i default -t 5 test.wav

# Play
ffplay test.wav
```

## Alternative Tools

### arecord (ALSA)

```bash
arecord -d 5 -f cd test.wav
aplay test.wav
```

### parecord (PipeWire/PulseAudio)

```bash
timeout 5 parecord test.wav
paplay test.wav
```

## List Devices

```bash
# ALSA devices
arecord -l

# PipeWire/PulseAudio sources
pactl list sources short
```

## Troubleshooting

- If no audio captured, check microphone is not muted
- Run `pavucontrol` to verify input device selection
- Check permissions: user should be in `audio` group
