/*
  ==============================================================================

    This file contains the basic framework code for a JUCE plugin processor.

  ==============================================================================
*/

#include "PluginProcessor.h"

//==============================================================================
SpectralBlendAudioProcessor::SpectralBlendAudioProcessor()
#ifndef JucePlugin_PreferredChannelConfigurations
     : AudioProcessor (BusesProperties()
                       // 4 input channels: Source A (L/R) + Source B (L/R)
                       .withInput  ("Input",  juce::AudioChannelSet::quadraphonic(), true)
                       // Stereo output: blended result
                       .withOutput ("Output", juce::AudioChannelSet::stereo(), true)
                       ),
       apvts(*this, nullptr, "Parameters", createParameters())
#endif
{
    apvts.state.addListener(this);
}

SpectralBlendAudioProcessor::~SpectralBlendAudioProcessor()
{
}

//==============================================================================
const juce::String SpectralBlendAudioProcessor::getName() const
{
    return JucePlugin_Name;
}

bool SpectralBlendAudioProcessor::acceptsMidi() const
{
   #if JucePlugin_WantsMidiInput
    return true;
   #else
    return false;
   #endif
}

bool SpectralBlendAudioProcessor::producesMidi() const
{
   #if JucePlugin_ProducesMidiOutput
    return true;
   #else
    return false;
   #endif
}

bool SpectralBlendAudioProcessor::isMidiEffect() const
{
   #if JucePlugin_IsMidiEffect
    return true;
   #else
    return false;
   #endif
}

double SpectralBlendAudioProcessor::getTailLengthSeconds() const
{
    return static_cast<double>(currentFFTSize) / currentSampleRate;
}

int SpectralBlendAudioProcessor::getNumPrograms()
{
    return 1;
}

int SpectralBlendAudioProcessor::getCurrentProgram()
{
    return 0;
}

void SpectralBlendAudioProcessor::setCurrentProgram (int index)
{
    juce::ignoreUnused(index);
}

const juce::String SpectralBlendAudioProcessor::getProgramName (int index)
{
    juce::ignoreUnused(index);
    return {};
}

void SpectralBlendAudioProcessor::changeProgramName (int index, const juce::String& newName)
{
    juce::ignoreUnused(index, newName);
}

//==============================================================================
void SpectralBlendAudioProcessor::prepareToPlay(double sampleRate, int samplesPerBlock)
{
    juce::ignoreUnused(samplesPerBlock);

    currentSampleRate = sampleRate;
    isActive = true;

    // Get FFT size from parameter (choice index: 0=512, 1=1024, 2=2048, 3=4096)
    auto fftSizeParam = apvts.getRawParameterValue("FFT_SIZE");
    int fftSizeIndex = static_cast<int>(fftSizeParam->load());
    static const int fftSizes[] = {512, 1024, 2048, 4096};
    currentFFTSize = fftSizes[std::clamp(fftSizeIndex, 0, 3)];
    currentWindowSize = currentFFTSize;
    currentHopSize = currentFFTSize / kDefaultHopDivisor;

    initSpectralProcessing();

    // Initialize smoothed values
    blendValue.reset(sampleRate, 0.05);  // 50ms smoothing
    outputGain.reset(sampleRate, 0.05);

    updateParameters();
}

void SpectralBlendAudioProcessor::initSpectralProcessing()
{
    // Create AudioTransport instances with the actual FFT size we'll use
    // Note: AudioTransport has a bug in init() when using sizes different from constructor
    fluid::Allocator& alloc = fluid::FluidDefaultAllocator();

    audioTransportL = std::make_unique<fluid::algorithm::AudioTransport>(currentFFTSize, alloc);
    audioTransportR = std::make_unique<fluid::algorithm::AudioTransport>(currentFFTSize, alloc);

    // Initialize the AudioTransport algorithms
    audioTransportL->init(currentWindowSize, currentFFTSize, currentHopSize);
    audioTransportR->init(currentWindowSize, currentFFTSize, currentHopSize);

    // Size the ring buffers - need at least 2x FFT size for overlap-add
    const int bufferSize = currentFFTSize * 4;

    for (auto& buf : inputBuffers)
    {
        buf.resize(static_cast<size_t>(bufferSize), 0.0);
    }

    for (int ch = 0; ch < 2; ++ch)
    {
        outputBuffers[static_cast<size_t>(ch)].resize(static_cast<size_t>(bufferSize), 0.0);
        windowAccumulator[static_cast<size_t>(ch)].resize(static_cast<size_t>(bufferSize), 0.0);
    }

    // Reset positions
    inputWritePos = 0;
    outputWritePos = currentFFTSize;  // Start output ahead by one FFT frame for latency compensation
    outputReadPos = 0;

    // Wait until we have accumulated at least one full window before processing
    samplesUntilNextHop = currentWindowSize;

    // Clear all buffers
    for (auto& buf : inputBuffers)
    {
        std::fill(buf.begin(), buf.end(), 0.0);
    }
    for (int ch = 0; ch < 2; ++ch)
    {
        std::fill(outputBuffers[static_cast<size_t>(ch)].begin(),
                  outputBuffers[static_cast<size_t>(ch)].end(), 0.0);
        std::fill(windowAccumulator[static_cast<size_t>(ch)].begin(),
                  windowAccumulator[static_cast<size_t>(ch)].end(), 0.0);
    }
}

void SpectralBlendAudioProcessor::releaseResources()
{
    audioTransportL.reset();
    audioTransportR.reset();
}

#ifndef JucePlugin_PreferredChannelConfigurations
bool SpectralBlendAudioProcessor::isBusesLayoutSupported (const BusesLayout& layouts) const
{
    // We need exactly 4 input channels and 2 output channels
    if (layouts.getMainInputChannelSet() != juce::AudioChannelSet::quadraphonic())
        return false;

    if (layouts.getMainOutputChannelSet() != juce::AudioChannelSet::stereo())
        return false;

    return true;
}
#endif

void SpectralBlendAudioProcessor::updateParameters()
{
    auto blend = apvts.getRawParameterValue("BLEND");
    auto gain = apvts.getRawParameterValue("OUTPUT_GAIN");

    blendValue.setTargetValue(blend->load());
    outputGain.setTargetValue(juce::Decibels::decibelsToGain(gain->load()));

    mustUpdateProcessing = false;
}

void SpectralBlendAudioProcessor::processSpectralFrame()
{
    if (!audioTransportL || !audioTransportR || !audioTransportL->initialized())
        return;

    fluid::Allocator& alloc = fluid::FluidDefaultAllocator();

    const int bufferSize = static_cast<int>(inputBuffers[0].size());

    // Calculate the read position for the frame (one window back from write position)
    int frameStart = (inputWritePos - currentWindowSize + bufferSize) % bufferSize;

    // Create FluidTensor views for the input frames - use the allocator
    fluid::index windowSize = static_cast<fluid::index>(currentWindowSize);
    fluid::RealVector frameA_L(windowSize, alloc);
    fluid::RealVector frameA_R(windowSize, alloc);
    fluid::RealVector frameB_L(windowSize, alloc);
    fluid::RealVector frameB_R(windowSize, alloc);

    // Copy data from ring buffers to frame buffers
    for (int i = 0; i < currentWindowSize; ++i)
    {
        int idx = (frameStart + i) % bufferSize;
        frameA_L(i) = inputBuffers[0][static_cast<size_t>(idx)];
        frameA_R(i) = inputBuffers[1][static_cast<size_t>(idx)];
        frameB_L(i) = inputBuffers[2][static_cast<size_t>(idx)];
        frameB_R(i) = inputBuffers[3][static_cast<size_t>(idx)];
    }

    // Output is 2 rows: [0] = audio output, [1] = window squared (for normalization)
    fluid::RealMatrix outputL(2, windowSize, alloc);
    fluid::RealMatrix outputR(2, windowSize, alloc);

    // Initialize output matrices to zero
    for (fluid::index r = 0; r < 2; ++r)
    {
        for (fluid::index c = 0; c < windowSize; ++c)
        {
            outputL(r, c) = 0.0;
            outputR(r, c) = 0.0;
        }
    }

    // Get current blend value
    double blend = static_cast<double>(blendValue.getCurrentValue());

    // Process frames through AudioTransport
    audioTransportL->processFrame(frameA_L, frameB_L, blend, outputL, alloc);
    audioTransportR->processFrame(frameA_R, frameB_R, blend, outputR, alloc);

    // Overlap-add the output
    for (int i = 0; i < currentWindowSize; ++i)
    {
        int outIdx = (outputWritePos + i) % bufferSize;

        outputBuffers[0][static_cast<size_t>(outIdx)] += outputL(0, i);
        outputBuffers[1][static_cast<size_t>(outIdx)] += outputR(0, i);

        windowAccumulator[0][static_cast<size_t>(outIdx)] += outputL(1, i);
        windowAccumulator[1][static_cast<size_t>(outIdx)] += outputR(1, i);
    }

    // Advance output write position by hop size
    outputWritePos = (outputWritePos + currentHopSize) % bufferSize;
}

void SpectralBlendAudioProcessor::processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midiMessages)
{
    juce::ignoreUnused(midiMessages);

    if (!isActive)
        return;

    if (mustUpdateProcessing)
        updateParameters();

    juce::ScopedNoDenormals noDenormals;

    const int numInputChannels = getTotalNumInputChannels();
    const int numOutputChannels = getTotalNumOutputChannels();
    const int numSamples = buffer.getNumSamples();

    // Need at least 4 input channels
    if (numInputChannels < 4)
    {
        buffer.clear();
        return;
    }

    const int bufferSize = static_cast<int>(inputBuffers[0].size());

    // Get input channel pointers
    const float* inputA_L = buffer.getReadPointer(0);
    const float* inputA_R = buffer.getReadPointer(1);
    const float* inputB_L = buffer.getReadPointer(2);
    const float* inputB_R = buffer.getReadPointer(3);

    // Get output channel pointers
    float* outputL = buffer.getWritePointer(0);
    float* outputR = numOutputChannels > 1 ? buffer.getWritePointer(1) : buffer.getWritePointer(0);

    for (int sample = 0; sample < numSamples; ++sample)
    {
        // Write input samples to ring buffers
        inputBuffers[0][static_cast<size_t>(inputWritePos)] = static_cast<double>(inputA_L[sample]);
        inputBuffers[1][static_cast<size_t>(inputWritePos)] = static_cast<double>(inputA_R[sample]);
        inputBuffers[2][static_cast<size_t>(inputWritePos)] = static_cast<double>(inputB_L[sample]);
        inputBuffers[3][static_cast<size_t>(inputWritePos)] = static_cast<double>(inputB_R[sample]);

        inputWritePos = (inputWritePos + 1) % bufferSize;

        // Process a frame when we've accumulated enough samples
        samplesUntilNextHop--;
        if (samplesUntilNextHop <= 0)
        {
            processSpectralFrame();
            samplesUntilNextHop = currentHopSize;

            // Update blend value for next frame
            blendValue.skip(currentHopSize);
        }

        // Read from output buffer with normalization
        double outL = 0.0;
        double outR = 0.0;

        double normL = windowAccumulator[0][static_cast<size_t>(outputReadPos)];
        double normR = windowAccumulator[1][static_cast<size_t>(outputReadPos)];

        constexpr double minNorm = 1e-10;

        if (normL > minNorm)
            outL = outputBuffers[0][static_cast<size_t>(outputReadPos)] / normL;
        if (normR > minNorm)
            outR = outputBuffers[1][static_cast<size_t>(outputReadPos)] / normR;

        // Clear the output buffer position for reuse
        outputBuffers[0][static_cast<size_t>(outputReadPos)] = 0.0;
        outputBuffers[1][static_cast<size_t>(outputReadPos)] = 0.0;
        windowAccumulator[0][static_cast<size_t>(outputReadPos)] = 0.0;
        windowAccumulator[1][static_cast<size_t>(outputReadPos)] = 0.0;

        outputReadPos = (outputReadPos + 1) % bufferSize;

        // Apply output gain
        float gain = outputGain.getNextValue();
        outputL[sample] = static_cast<float>(outL) * gain;
        outputR[sample] = static_cast<float>(outR) * gain;
    }
}

//==============================================================================
bool SpectralBlendAudioProcessor::hasEditor() const
{
    return true;
}

juce::AudioProcessorEditor* SpectralBlendAudioProcessor::createEditor()
{
    return new juce::GenericAudioProcessorEditor (*this);
    //return new SpectralBlendAudioProcessorEditor(*this);
}

//==============================================================================
void SpectralBlendAudioProcessor::getStateInformation (juce::MemoryBlock& destData)
{
    juce::ValueTree copyState = apvts.copyState();
    std::unique_ptr<juce::XmlElement> xml = copyState.createXml();
    copyXmlToBinary(*xml.get(), destData);
}

void SpectralBlendAudioProcessor::setStateInformation (const void* data, int sizeInBytes)
{
    std::unique_ptr<juce::XmlElement> xml = getXmlFromBinary(data, sizeInBytes);
    if (xml)
    {
        juce::ValueTree copyState = juce::ValueTree::fromXml(*xml.get());
        apvts.replaceState(copyState);
    }
}

juce::AudioProcessorValueTreeState::ParameterLayout SpectralBlendAudioProcessor::createParameters()
{
    std::vector<std::unique_ptr<juce::RangedAudioParameter>> parameters;

    // Blend parameter: 0 = 100% Source A, 1 = 100% Source B
    parameters.push_back(std::make_unique<juce::AudioParameterFloat>(
        "BLEND", "Blend",
        juce::NormalisableRange<float>(0.0f, 1.0f, 0.001f),
        0.5f,
        juce::String(),
        juce::AudioProcessorParameter::genericParameter,
        [](float value, int) { return juce::String(static_cast<int>(value * 100)) + "%"; },
        [](const juce::String& text) { return text.getFloatValue() / 100.0f; }
    ));

    // Output gain in dB
    // Output gain in dB
    parameters.push_back(std::make_unique<juce::AudioParameterFloat>(
        "OUTPUT_GAIN", "Output Gain",
        juce::NormalisableRange<float>(-24.0f, 12.0f, 0.1f),
        0.0f,
        "dB"
    ));

    // FFT Size parameter
    parameters.push_back(std::make_unique<juce::AudioParameterChoice>(
        "FFT_SIZE", "FFT Size",
        juce::StringArray{"512", "1024", "2048", "4096"},
        1  // Default to 1024
    ));

    return { parameters.begin(), parameters.end() };
}

//==============================================================================
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new SpectralBlendAudioProcessor();
}
