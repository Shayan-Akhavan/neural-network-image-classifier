import React, { useEffect, useState } from 'react';

const ShapeDisplay = () => {
  const [shapes, setShapes] = useState([]);
  
  useEffect(() => {
    // Generate 9 random shapes
    const shapeTypes = ['circle', 'square', 'triangle'];
    const newShapes = Array(9).fill(null).map((_, i) => ({
      type: shapeTypes[i % 3],
      color: `hsl(${Math.random() * 360}, 70%, 50%)`
    }));
    setShapes(newShapes);
  }, []);

  const renderShape = (shape, size = 100) => {
    switch (shape.type) {
      case 'circle':
        return (
          <svg width={size} height={size} viewBox="0 0 100 100">
            <circle cx="50" cy="50" r="40" fill={shape.color} />
          </svg>
        );
      case 'square':
        return (
          <svg width={size} height={size} viewBox="0 0 100 100">
            <rect x="10" y="10" width="80" height="80" fill={shape.color} />
          </svg>
        );
      case 'triangle':
        return (
          <svg width={size} height={size} viewBox="0 0 100 100">
            <polygon points="50,10 10,90 90,90" fill={shape.color} />
          </svg>
        );
      default:
        return null;
    }
  };

  return (
    <div className="p-4">
      <h2 className="text-xl font-bold mb-4">Training Images Preview</h2>
      <div className="grid grid-cols-3 gap-4">
        {shapes.map((shape, i) => (
          <div key={i} className="border p-2 rounded-lg">
            {renderShape(shape)}
            <p className="text-center mt-2">{shape.type}</p>
          </div>
        ))}
      </div>
    </div>
  );
};

export default ShapeDisplay;