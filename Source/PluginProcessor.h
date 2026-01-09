/*
  ==============================================================================

    This file contains the basic framework code for a JUCE plugin processor.

  ==============================================================================
*/

#pragma once

#include <juce_audio_basics/juce_audio_basics.h>
#include <juce_audio_processors/juce_audio_processors.h>
#include <flucoma/algorithms/public/AudioTransport.hpp>

//==============================================================================
/**
*/
class SpectralBlendAudioProcessor  : public juce::AudioProcessor,
                                     public juce::ValueTree::Listener
                            #if JucePlugin_Enable_ARA
                             , public juce::AudioProcessorARAExtension
                            #endif
{
public:
    //==============================================================================
    SpectralBlendAudioProcessor();
    ~SpectralBlendAudioProcessor() override;

    //==============================================================================
    void prepareToPlay (double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;

   #ifndef JucePlugin_PreferredChannelConfigurations
    bool isBusesLayoutSupported (const BusesLayout& layouts) const override;
   #endif

    void processBlock (juce::AudioBuffer<float>&, juce::MidiBuffer&) override;

    //==============================================================================
    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override;

    //==============================================================================
    const juce::String getName() const override;

    bool acceptsMidi() const override;
    bool producesMidi() const override;
    bool isMidiEffect() const override;
    double getTailLengthSeconds() const override;

    //==============================================================================
    int getNumPrograms() override;
    int getCurrentProgram() override;
    void setCurrentProgram (int index) override;
    const juce::String getProgramName (int index) override;
    void changeProgramName (int index, const juce::String& newName) override;

    //==============================================================================
    void getStateInformation (juce::MemoryBlock& destData) override;
    void setStateInformation (const void* data, int sizeInBytes) override;

    //==============================================================================
    // Store Parameters
    juce::AudioProcessorValueTreeState apvts;

private:
    juce::AudioProcessorValueTreeState::ParameterLayout createParameters();

    // Called when user changes a parameter
    void valueTreePropertyChanged(juce::ValueTree& tree, const juce::Identifier& property) override
    {
        mustUpdateProcessing = true;
    }

    void updateParameters();
    void initSpectralProcessing();
    void processSpectralFrame();

    bool isActive{ false };
    bool mustUpdateProcessing{ false };

    // FFT parameters
    static constexpr int kMaxFFTSize = 4096;
    static constexpr int kDefaultFFTSize = 1024;
    static constexpr int kDefaultHopDivisor = 4;  // hopSize = fftSize / 4

    int currentFFTSize{ kDefaultFFTSize };
    int currentHopSize{ kDefaultFFTSize / kDefaultHopDivisor };
    int currentWindowSize{ kDefaultFFTSize };

    // AudioTransport instances (one per channel for stereo processing)
    std::unique_ptr<fluid::algorithm::AudioTransport> audioTransportL;
    std::unique_ptr<fluid::algorithm::AudioTransport> audioTransportR;

    // Input ring buffers for all 4 channels
    // [0] = Source A Left, [1] = Source A Right, [2] = Source B Left, [3] = Source B Right
    std::array<std::vector<double>, 4> inputBuffers;
    int inputWritePos{ 0 };

    // Output ring buffers with overlap-add (stereo)
    std::array<std::vector<double>, 2> outputBuffers;
    std::array<std::vector<double>, 2> windowAccumulator;
    int outputWritePos{ 0 };
    int outputReadPos{ 0 };

    // Processing state
    int samplesUntilNextHop{ 0 };
    double currentSampleRate{ 44100.0 };

    // Smoothed parameter values
    juce::LinearSmoothedValue<float> blendValue{ 0.5f };
    juce::LinearSmoothedValue<float> outputGain{ 1.0f };

    //==============================================================================
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (SpectralBlendAudioProcessor)
};
