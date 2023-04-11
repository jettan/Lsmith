export enum Scheduler {
    'DDIM' = 'ddim',
    'DPM++' = 'dpm++',
    'Euler A' = 'euler_a',
    'LMSD' = 'lmsd',
    'PNDM' = 'pndm',
}

export type SchedulerName = keyof typeof Scheduler
export const schedulerNames: SchedulerName[] = Object.keys(Scheduler) as SchedulerName[]

export const categoryList = ['txt2img', 'img2img'] as const
export type categoryType = (typeof categoryList)[number]
