import {controls,mix,clamp} from './kin.js';
export const duration=25;
export const beats=[
 {at:0,end:4,preset:'open',view:'solid',focus:'all',camera:'front',title:'01 / Architecture',note:'Five fingers, 20 modeled joint axes. The five flexor cables share a compact drive block.'},
 {at:4,end:8,preset:'power',view:'solid',focus:'all',camera:'three',title:'02 / Power grasp',note:'Cable take-up curls the linked joints. Grip is a prescribed posture, not a contact solve.'},
 {at:8,end:12,preset:'power',view:'tendon',focus:'index',camera:'detail',title:'03 / Tendon routing',note:'The teal flexor wraps three guides. Pull length is the sum of moment arm times joint rotation.'},
 {at:12,end:16,preset:'pinch',view:'tendon',focus:'all',camera:'front',title:'04 / Thumb opposition',note:'Rotation at the thumb base brings the tip toward the index. The line measures tip-center distance.'},
 {at:16,end:20,preset:'open',view:'exploded',focus:'all',camera:'three',title:'05 / Exploded covers',note:'Lift the covers to expose rails, hinge pins, five capstans and schematic cable routing.'},
 {at:20,end:25,preset:'tripod',view:'skeleton',focus:'all',camera:'three',title:'06 / Frame & control',note:'The structure and motion remain identical when the outer cover is hidden.'}
];
export function tour(t){t=((t%duration)+duration)%duration;let i=beats.findIndex(b=>t>=b.at&&t<b.end);if(i<0)i=0;const b=beats[i],prev=beats[(i+beats.length-1)%beats.length];let x=clamp((t-b.at)/1.35);x=x*x*(3-2*x);return {...b,c:i===0?controls('open'):mix(controls(prev.preset),controls(b.preset),x),progress:t/duration,index:i};}
