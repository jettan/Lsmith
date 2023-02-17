/* tslint:disable */
/* eslint-disable */
/**
 * FastAPI
 * No description provided (generated by Openapi Generator https://github.com/openapitools/openapi-generator)
 *
 * The version of the OpenAPI document: 0.1.0
 * 
 *
 * NOTE: This class is auto generated by OpenAPI Generator (https://openapi-generator.tech).
 * https://openapi-generator.tech
 * Do not edit the class manually.
 */

import { exists, mapValues } from '../runtime';
import type { ImageGenerationProgress } from './ImageGenerationProgress';
import {
    ImageGenerationProgressFromJSON,
    ImageGenerationProgressFromJSONTyped,
    ImageGenerationProgressToJSON,
} from './ImageGenerationProgress';
import type { ImageGenerationResult } from './ImageGenerationResult';
import {
    ImageGenerationResultFromJSON,
    ImageGenerationResultFromJSONTyped,
    ImageGenerationResultToJSON,
} from './ImageGenerationResult';
import type { ImageInformation } from './ImageInformation';
import {
    ImageInformationFromJSON,
    ImageInformationFromJSONTyped,
    ImageInformationToJSON,
} from './ImageInformation';

/**
 * 
 * @export
 * @interface Data
 */
export interface Data {
    /**
     * 
     * @type {string}
     * @memberof Data
     */
    type?: DataTypeEnum;
    /**
     * 
     * @type {{ [key: string]: ImageInformation; }}
     * @memberof Data
     */
    images: { [key: string]: ImageInformation; };
    /**
     * 
     * @type {number}
     * @memberof Data
     */
    performance: number;
    /**
     * 
     * @type {number}
     * @memberof Data
     */
    progress: number;
}


/**
 * @export
 */
export const DataTypeEnum = {
    progress: 'progress'
} as const;
export type DataTypeEnum = typeof DataTypeEnum[keyof typeof DataTypeEnum];


/**
 * Check if a given object implements the Data interface.
 */
export function instanceOfData(value: object): boolean {
    let isInstance = true;
    isInstance = isInstance && "images" in value;
    isInstance = isInstance && "performance" in value;
    isInstance = isInstance && "progress" in value;

    return isInstance;
}

export function DataFromJSON(json: any): Data {
    return DataFromJSONTyped(json, false);
}

export function DataFromJSONTyped(json: any, ignoreDiscriminator: boolean): Data {
    if ((json === undefined) || (json === null)) {
        return json;
    }
    return {
        
        'type': !exists(json, 'type') ? undefined : json['type'],
        'images': (mapValues(json['images'], ImageInformationFromJSON)),
        'performance': json['performance'],
        'progress': json['progress'],
    };
}

export function DataToJSON(value?: Data | null): any {
    if (value === undefined) {
        return undefined;
    }
    if (value === null) {
        return null;
    }
    return {
        
        'type': value.type,
        'images': (mapValues(value.images, ImageInformationToJSON)),
        'performance': value.performance,
        'progress': value.progress,
    };
}
