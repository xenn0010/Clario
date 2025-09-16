import { Navigate, Route, Routes } from 'react-router-dom';
import BaseLayout from './layouts/BaseLayout';
import Landing from './pages/Landing';
import Dashboard from './pages/Dashboard';
import GraphStudio from './pages/GraphStudio';

const App = () => {
  return (
    <BaseLayout>
      <Routes>
        <Route path="/" element={<Landing />} />
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/graph-studio" element={<GraphStudio />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BaseLayout>
  );
};

export default App;
