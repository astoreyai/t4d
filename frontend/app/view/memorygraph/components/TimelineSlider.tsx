/**
 * Timeline slider for replaying memory formation over time
 */

import React from "react";
import { useAtom } from "jotai";
import { timelineStateAtom, updateTimelineAtom, toggleTimelinePlaybackAtom } from "../memorygraph-state";
import { TimelineSliderProps } from "../memorygraph-types";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { Label } from "@/components/ui/label";
import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue,
} from "@/components/ui/select";
import "./TimelineSlider.scss";

/**
 * Format timestamp to readable date
 */
const formatDate = (timestamp: number): string => {
    const date = new Date(timestamp);
    return date.toLocaleDateString("en-US", {
        month: "short",
        day: "numeric",
        year: "numeric",
        hour: "2-digit",
        minute: "2-digit",
    });
};

/**
 * Timeline slider component
 */
export const TimelineSlider: React.FC<TimelineSliderProps> = ({
    timelineState,
    onTimeChange,
    onPlayPause,
    onSpeedChange,
}) => {
    const [timeline, setTimeline] = useAtom(timelineStateAtom);
    const updateTimeline = useAtom(updateTimelineAtom)[1];
    const togglePlayback = useAtom(toggleTimelinePlaybackAtom)[1];

    const handleTimeChange = (values: number[]) => {
        const newTime = values[0];
        updateTimeline({ currentTime: newTime });
        onTimeChange?.(newTime);
    };

    const handlePlayPause = () => {
        togglePlayback();
        onPlayPause?.();
    };

    const handleSpeedChange = (value: string) => {
        const speed = parseFloat(value);
        updateTimeline({ playbackSpeed: speed });
        onSpeedChange?.(speed);
    };

    const handleSkipToStart = () => {
        updateTimeline({ currentTime: timeline.startTime });
    };

    const handleSkipToEnd = () => {
        updateTimeline({ currentTime: timeline.endTime });
    };

    const progress =
        ((timeline.currentTime - timeline.startTime) / (timeline.endTime - timeline.startTime)) * 100;

    return (
        <div className="timeline-slider">
            <Card>
                <CardContent className="timeline-content">
                    {/* Time Display */}
                    <div className="time-display">
                        <div className="time-label">
                            <Label>Current Time</Label>
                            <span className="time-value">{formatDate(timeline.currentTime)}</span>
                        </div>
                        <div className="time-range">
                            <span className="range-start">{formatDate(timeline.startTime)}</span>
                            <span className="range-end">{formatDate(timeline.endTime)}</span>
                        </div>
                    </div>

                    {/* Progress Bar */}
                    <div className="progress-container">
                        <div className="progress-bar">
                            <div className="progress-fill" style={{ width: `${progress}%` }} />
                        </div>
                    </div>

                    {/* Time Slider */}
                    <div className="slider-container">
                        <Slider
                            value={[timeline.currentTime]}
                            min={timeline.startTime}
                            max={timeline.endTime}
                            step={1000 * 60 * 60} // 1 hour steps
                            onValueChange={handleTimeChange}
                            className="time-slider"
                        />
                    </div>

                    {/* Controls */}
                    <div className="timeline-controls">
                        {/* Playback Buttons */}
                        <div className="playback-buttons">
                            <Button
                                variant="outline"
                                size="sm"
                                onClick={handleSkipToStart}
                                disabled={timeline.currentTime === timeline.startTime}
                            >
                                <i className="fa fa-backward-step"></i>
                            </Button>

                            <Button
                                variant="default"
                                size="sm"
                                onClick={handlePlayPause}
                                className="play-button"
                            >
                                {timeline.isPlaying ? (
                                    <i className="fa fa-pause"></i>
                                ) : (
                                    <i className="fa fa-play"></i>
                                )}
                            </Button>

                            <Button
                                variant="outline"
                                size="sm"
                                onClick={handleSkipToEnd}
                                disabled={timeline.currentTime === timeline.endTime}
                            >
                                <i className="fa fa-forward-step"></i>
                            </Button>
                        </div>

                        {/* Speed Control */}
                        <div className="speed-control">
                            <Label>Speed</Label>
                            <Select
                                value={timeline.playbackSpeed.toString()}
                                onValueChange={handleSpeedChange}
                            >
                                <SelectTrigger className="speed-select">
                                    <SelectValue />
                                </SelectTrigger>
                                <SelectContent>
                                    <SelectItem value="0.25">0.25x</SelectItem>
                                    <SelectItem value="0.5">0.5x</SelectItem>
                                    <SelectItem value="1">1x</SelectItem>
                                    <SelectItem value="2">2x</SelectItem>
                                    <SelectItem value="5">5x</SelectItem>
                                    <SelectItem value="10">10x</SelectItem>
                                </SelectContent>
                            </Select>
                        </div>
                    </div>
                </CardContent>
            </Card>
        </div>
    );
};
