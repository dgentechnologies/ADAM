'use client';

import React, { useEffect, useRef } from 'react';
import { cn } from '@adam/ui';

interface CanvasRevealEffectProps {
  animationSpeed?: number;
  opacities?: number[];
  colors?: number[][];
  containerClassName?: string;
  dotSize?: number;
  showGradient?: boolean;
  reverse?: boolean;
}

export function CanvasRevealEffect({
  animationSpeed = 3,
  opacities = [0.3, 0.3, 0.3, 0.5, 0.5, 0.5, 0.8, 0.8, 0.8, 1],
  colors = [[255, 255, 255], [255, 255, 255]],
  containerClassName,
  dotSize = 4,
  showGradient = true,
  reverse = false,
}: CanvasRevealEffectProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  // Callers pass inline arrays; typing into a form must not rebuild its GPU scene.
  const colorsKey = JSON.stringify(colors);
  const opacitiesKey = JSON.stringify(opacities);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const colors: number[][] = JSON.parse(colorsKey);
    const opacities: number[] = JSON.parse(opacitiesKey);

    const gl = canvas.getContext('webgl2');
    if (!gl) return;

    const vsSource = `#version 300 es
      precision mediump float;
      in vec2 position;
      out vec2 fragCoord;
      uniform vec2 u_resolution;
      void main() {
        gl_Position = vec4(position, 0.0, 1.0);
        fragCoord = (position + vec2(1.0)) * 0.5 * u_resolution;
        fragCoord.y = u_resolution.y - fragCoord.y;
      }
    `;

    const fsSource = `#version 300 es
      precision mediump float;
      in vec2 fragCoord;
      out vec4 fragColor;

      uniform float u_time;
      uniform float u_opacities[10];
      uniform vec3 u_colors[6];
      uniform float u_total_size;
      uniform float u_dot_size;
      uniform vec2 u_resolution;
      uniform int u_reverse;

      float PHI = 1.61803398874989484820459;
      float random(vec2 xy) {
          return fract(tan(distance(xy * PHI, xy) * 0.5) * xy.x);
      }

      void main() {
          vec2 st = fragCoord.xy;
          st.x -= abs(floor((mod(u_resolution.x, u_total_size) - u_dot_size) * 0.5));
          st.y -= abs(floor((mod(u_resolution.y, u_total_size) - u_dot_size) * 0.5));

          float opacity = step(0.0, st.x);
          opacity *= step(0.0, st.y);

          vec2 st2 = vec2(int(st.x / u_total_size), int(st.y / u_total_size));

          float frequency = 5.0;
          float show_offset = random(st2);
          float rand = random(st2 * floor((u_time / frequency) + show_offset + frequency));
          int opIndex = clamp(int(rand * 10.0), 0, 9);
          opacity *= u_opacities[opIndex];
          opacity *= 1.0 - step(u_dot_size / u_total_size, fract(st.x / u_total_size));
          opacity *= 1.0 - step(u_dot_size / u_total_size, fract(st.y / u_total_size));

          int colIndex = clamp(int(show_offset * 6.0), 0, 5);
          vec3 color = u_colors[colIndex];

          float animation_speed_factor = 0.5;
          vec2 center_grid = u_resolution / 2.0 / u_total_size;
          float dist_from_center = distance(center_grid, st2);

          float timing_offset_intro = dist_from_center * 0.01 + (random(st2) * 0.15);
          float max_grid_dist = distance(center_grid, vec2(0.0, 0.0));
          float timing_offset_outro = (max_grid_dist - dist_from_center) * 0.02 + (random(st2 + 42.0) * 0.2);

          float current_timing_offset;
          if (u_reverse == 1) {
              current_timing_offset = timing_offset_outro;
              opacity *= 1.0 - step(current_timing_offset, u_time * animation_speed_factor);
              opacity *= clamp((step(current_timing_offset + 0.1, u_time * animation_speed_factor)) * 1.25, 1.0, 1.25);
          } else {
              // Instantaneous visibility: dots are visible and twinkling immediately on mount
              opacity *= clamp(u_time * 8.0 + 0.8, 0.0, 1.0);
          }

          fragColor = vec4(color, opacity);
          fragColor.rgb *= fragColor.a;
      }
    `;

    function createShader(glCtx: WebGL2RenderingContext, type: number, source: string) {
      const shader = glCtx.createShader(type);
      if (!shader) return null;
      glCtx.shaderSource(shader, source);
      glCtx.compileShader(shader);
      if (!glCtx.getShaderParameter(shader, glCtx.COMPILE_STATUS)) {
        console.error('Shader compile error:', glCtx.getShaderInfoLog(shader));
        glCtx.deleteShader(shader);
        return null;
      }
      return shader;
    }

    const vs = createShader(gl, gl.VERTEX_SHADER, vsSource);
    const fs = createShader(gl, gl.FRAGMENT_SHADER, fsSource);
    if (!vs || !fs) {
      if (vs) gl.deleteShader(vs);
      if (fs) gl.deleteShader(fs);
      return;
    }

    const program = gl.createProgram();
    if (!program) { gl.deleteShader(vs); gl.deleteShader(fs); return; }
    gl.attachShader(program, vs);
    gl.attachShader(program, fs);
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      console.warn('Program link error:', gl.getProgramInfoLog(program));
      gl.deleteProgram(program);
      gl.deleteShader(vs);
      gl.deleteShader(fs);
      return;
    }

    gl.useProgram(program);

    const positionBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
    gl.bufferData(
      gl.ARRAY_BUFFER,
      new Float32Array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]),
      gl.STATIC_DRAW
    );

    const posAttr = gl.getAttribLocation(program, 'position');
    gl.enableVertexAttribArray(posAttr);
    gl.vertexAttribPointer(posAttr, 2, gl.FLOAT, false, 0, 0);

    const uTimeLoc = gl.getUniformLocation(program, 'u_time');
    const uResolutionLoc = gl.getUniformLocation(program, 'u_resolution');
    const uTotalSizeLoc = gl.getUniformLocation(program, 'u_total_size');
    const uDotSizeLoc = gl.getUniformLocation(program, 'u_dot_size');
    const uReverseLoc = gl.getUniformLocation(program, 'u_reverse');
    const uOpacitiesLoc = gl.getUniformLocation(program, 'u_opacities');
    const uColorsLoc = gl.getUniformLocation(program, 'u_colors');

    gl.uniform1f(uTotalSizeLoc, 20.0);
    gl.uniform1f(uDotSizeLoc, dotSize);
    gl.uniform1i(uReverseLoc, reverse ? 1 : 0);
    gl.uniform1fv(uOpacitiesLoc, new Float32Array(opacities));

    const colorsFlattened = new Float32Array(18);
    for (let i = 0; i < 6; i++) {
      const c = (colors.length > 0 ? colors[i % colors.length] : undefined) ?? [255, 255, 255];
      colorsFlattened[i * 3 + 0] = (c[0] ?? 255) / 255;
      colorsFlattened[i * 3 + 1] = (c[1] ?? 255) / 255;
      colorsFlattened[i * 3 + 2] = (c[2] ?? 255) / 255;
    }
    gl.uniform3fv(uColorsLoc, colorsFlattened);

    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE);

    let animId = 0;
    const startTime = performance.now();
    let lastFrame = -Infinity;
    const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');

    const resize = () => {
      const dpr = Math.min(window.devicePixelRatio || 1, 1.5);
      const width = Math.round(canvas.clientWidth * dpr);
      const height = Math.round(canvas.clientHeight * dpr);
      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
        gl.viewport(0, 0, width, height);
        gl.uniform2f(uResolutionLoc, width, height);
      }
    };

    const render = (time: number) => {
      if (document.hidden) return;
      if (time - lastFrame < 1000 / 30 && !reducedMotion.matches) {
        animId = requestAnimationFrame(render);
        return;
      }
      lastFrame = time;
      resize();
      const elapsed = reducedMotion.matches ? 1 : ((time - startTime) / 1000) * (animationSpeed / 2);
      gl.uniform1f(uTimeLoc, elapsed);
      gl.drawArrays(gl.TRIANGLES, 0, 6);
      if (!reducedMotion.matches) animId = requestAnimationFrame(render);
    };

    const resume = () => {
      cancelAnimationFrame(animId);
      lastFrame = -Infinity;
      if (!document.hidden) animId = requestAnimationFrame(render);
    };
    resume();

    window.addEventListener('resize', resume);
    document.addEventListener('visibilitychange', resume);
    reducedMotion.addEventListener('change', resume);
    return () => {
      cancelAnimationFrame(animId);
      window.removeEventListener('resize', resume);
      document.removeEventListener('visibilitychange', resume);
      reducedMotion.removeEventListener('change', resume);
      gl.deleteBuffer(positionBuffer);
      gl.deleteProgram(program);
      gl.deleteShader(vs);
      gl.deleteShader(fs);
    };
  }, [animationSpeed, colorsKey, dotSize, opacitiesKey, reverse]);

  return (
    <div className={cn('relative h-full w-full pointer-events-none', containerClassName)}>
      <canvas ref={canvasRef} className="absolute inset-0 h-full w-full" />
      {showGradient && (
        <div className="absolute inset-0 bg-gradient-to-t from-black to-transparent pointer-events-none" />
      )}
    </div>
  );
}
