package com.yourorg.scheduler.config;

import java.util.List;

/**
 * The input files contain something the run must not continue with (for example trip_list problems under
 * {@code trip_validation: strict}). The message always starts with {@code "InvalidInputData: "}.
 */
public class InvalidInputDataException extends RuntimeException {

    public static final String PREFIX = "InvalidInputData: ";

    private final List<String> details;

    public InvalidInputDataException(String message) {
        this(message, List.of());
    }

    /** @param details the individual problems (for example the trip warnings) behind the message */
    public InvalidInputDataException(String message, List<String> details) {
        super(message.startsWith(PREFIX) ? message : PREFIX + message);
        this.details = List.copyOf(details);
    }

    public List<String> details() {
        return details;
    }
}
